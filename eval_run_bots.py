# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#%%

import json
import time

try:
    from eval_server_common import connect_to_redis
except ImportError:
    print("HINT: copy example.eval_server_common.py to eval_server_common.py")
    raise
import codraw_data
import episode

#%%

FAREWELL_MSG = "that's it, thanks!"

class Bot():
    model_name = "model_generic"
    agent_type = None
    fns = None

    # TODO(nikita): peek action for bot drawers is not supported
    def __init__(self, id):
        self.id = id
        self.episode = episode.Episode()
        self.role = "question" if self.agent_type == codraw_data.Agent.DRAWER else "answer"
        self.handlers = {
            'paired': self.on_paired,
            'receive message': self.on_receive_message,
            'server error': self.on_server_error, #TODO(nikita): Not emitted after I modified the server code
            'disconnected partner': self.on_disconnected_partner,
        }
        self.disconnected = False

        self.num_messages_sent = 0

    def disconnect(self):
        if self.id in type(self).active_bots:
            assert type(self).active_bots[self.id] == self
            del type(self).active_bots[self.id]

        if not self.disconnected:
            self.disconnected = True
            self.emit('disconnect')

    def emit(self, event, **msg):
        obj = {
            'botId': self.id,
            'event': event,
            'msg': msg,
        }
        self.redis.publish('visdial_server', json.dumps(obj))

    def send_msg(self, msg):
        self.num_messages_sent += 1
        self.emit('chat message', msg=msg, role=self.role, seqId=self.num_messages_sent)
        print("Sent chat message:", msg)

    def send_scene_log(self, scene):
        self.emit('scene log', scene=scene.stringify(), role=self.role, seqId=self.num_messages_sent)

    # TODO(nikita): implement drawer bots, including "send_scene_log" which is sent by drawer
    # socket.emit('scene log', {scene: Abs.resultAMT(), hitId: hitId, assignmentId: assignmentId, workerId: workerId, role: workerRole, seqId: noOfMsg});

    def run_model_actions(self, must_trigger=True):
        old_len = len(self.episode)
        terminated = self._run_model_actions()
        if terminated:
            print("No action taking. Disconnecting")
            if INTERACTIVE:
                display(self.episode.get_true_scene())
            self.send_msg(FAREWELL_MSG)
            self.disconnect()
            return

        if must_trigger:
            if len(self.episode) == old_len:
                self.disconnect()
                assert False, f"No response for event: {type(self.episode[-1]).__name__}"

        msg_to_send = None
        do_send_scene_log = False
        for event in self.episode[old_len:]:
            # TODO(nikita): log latent actions, such as SelectClipart
            if isinstance(event, codraw_data.TellGroup):
                assert msg_to_send is None, "Multiple TellGroup events added in a single round!"
                msg_to_send = event.msg
            elif isinstance(event, codraw_data.ReplyGroup):
                assert msg_to_send is None, "Multiple ReplyGroup events added in a single round!"
                msg_to_send = event.msg
            elif isinstance(event, (codraw_data.DrawClipart, codraw_data.DrawGroup)):
                do_send_scene_log = True

        if do_send_scene_log:
            assert self.agent_type == codraw_data.Agent.DRAWER
            self.send_scene_log(self.episode.reconstruct())

        if self.agent_type == codraw_data.Agent.TELLER:
            assert msg_to_send is not None, "No message to send"
            # Empty message is a signal for the drawer to begin the conversation
            if msg_to_send == "" and len([x for x in self.episode if isinstance(x, codraw_data.TellGroup)]) == 1:
                msg_to_send = None
                print("Model expects the human drawer to start the conversation.")
        else:
            assert msg_to_send is not None or isinstance(self.episode[-1], codraw_data.ObserveTruth), "No message to send, and not the start"

        if msg_to_send is not None:
            self.send_msg(msg_to_send)

    def _run_model_actions(self):
        while True:
            for fn in self.fns:
                if type(self.episode[-1]) in fn._trigger_types:
                    old_len = len(self.episode)
                    fn(self.episode)
                    if len(self.episode) == old_len:
                        return True # terminated
                    break
            else:
                # print('no trigger for', type(self.episode[-1]))
                return False

    def on_paired(self, partnerId=None, key=None, image_url=None, role=None, caption=None):
        if self.disconnected:
            print("[ERROR] Disconnected bot was paired!")
            return
        print("Paired wih human partner!")
        print("image_url:", image_url)
        print("partner role:", role) # Yes, the role sent in the message is for the partner
        assigned_role = "question" if role == "answer" else "answer"
        assert assigned_role == self.role, "Wrong role assigned to bot!"

        true_scene = codraw_data.AbstractScene(image_url)
        self.episode.append(codraw_data.ObserveTruth(true_scene))
        self.run_model_actions(must_trigger=False)

    def on_receive_message(self, message=None, noOfMsg=None):
        if self.disconnected:
            print("[ERROR] Disconnected bot received a message!")
            return
        print(f"Got human message {noOfMsg}: {message}")
        assert message is not None

        if self.agent_type == codraw_data.Agent.TELLER:
            self.episode.append(codraw_data.ReplyGroup(message))
        else:
            self.episode.append(codraw_data.TellGroup(message))
        self.run_model_actions()

    def on_disconnected_partner(self, disable='_unused'):
        print("Partner disconnected from bot! Cleanining up the bot")
        self.disconnect()

    def on_server_error(self, errorMsg='[no errorMsg specified]'):
        print("Error from server:", errorMsg)
        self.disconnect()

# %%


def run_loop(classes):
    active_bots = {}
    channel_to_cls = {}

    for cls in classes:
        assert cls.agent_type in (codraw_data.Agent.TELLER, codraw_data.Agent.DRAWER), "Invalid agent_type for bot!"

        channel = f'visdial_models.{cls.model_name}'.encode('utf-8')
        assert channel not in channel_to_cls, f"Duplicate model name {cls.model_name}"
        channel_to_cls[channel] = cls

        if not hasattr(cls, 'redis'):
            cls.redis = connect_to_redis()

        if not hasattr(cls, 'active_bots'):
            cls.active_bots = active_bots

    p = cls.redis.pubsub()

    for channel in channel_to_cls:
        p.subscribe(channel)

    for redis_msg in p.listen():
        print("Got redis msg", redis_msg)
        if redis_msg['type'] != 'message':
            continue

        if redis_msg['channel'] not in channel_to_cls:
            print(f"WARNING: unrecognized channel {redis_msg['channel']}")
            continue

        data = json.loads(redis_msg['data'])

        id = data['botId']
        event = data['event']
        msg = data['msg']

        if event == 'paired':
            active_bots[id] = channel_to_cls[redis_msg['channel']](id)

        if id in active_bots:
            handler = active_bots[id].handlers.get(event, None)
            if handler is None:
                print(f"No handler for event '{event}'")
            else:
                active_bots[id].handlers[event](**msg)

# %%

def make_script_teller_class():
    import model

    class ScriptTellerBot(Bot):
        model_name = 'teller_script'
        agent_type = codraw_data.Agent.TELLER
        fns = [model.scripted_tell_before_peek]
        scene_to_script = {}

        def _run_model_actions(self):
            if not hasattr(self.episode, 'script'):
                script = self.scene_to_script[self.episode.get_last(codraw_data.ObserveTruth).scene.stringify()]
                self.episode.script = script
                self.episode.script_index = 0
            return super()._run_model_actions()

    for scene, script in codraw_data.get_scenes_and_scripts('all'):
        ScriptTellerBot.scene_to_script[scene.stringify()] = script

    return ScriptTellerBot

# %%

def model_to_bot_class(model_name, model, model_agent_type=codraw_data.Agent.TELLER):
    model_name_ = model_name
    class TheBot(Bot):
        model_name = model_name_
        agent_type = model_agent_type
        fns = model.get_action_fns()

    TheBot.__name__ = type(model).__name__ + 'Bot'
    TheBot.__qualname__ = TheBot.__qualname__.replace('TheBot', TheBot.__name__)
    return TheBot

# %%

def run_model_pairs(tellers, drawers=[], include_script_teller=True):
    classes = []

    if include_script_teller:
        classes.append(make_script_teller_class())

    for teller_name, (a, b) in tellers:
        if a is not None:
            classes.append(model_to_bot_class(teller_name + '_a', a, codraw_data.Agent.TELLER))
        if b is not None:
            classes.append(model_to_bot_class(teller_name + '_b', b, codraw_data.Agent.TELLER))

    for drawer_name, (a, b) in drawers:
        if a is not None:
            classes.append(model_to_bot_class(drawer_name + '_a', a, codraw_data.Agent.DRAWER))
        if b is not None:
            classes.append(model_to_bot_class(drawer_name + '_b', b, codraw_data.Agent.DRAWER))

    run_loop(classes)

#%%

if __name__ == '__main__':
    from saved_models import load_models, make_pairs
    models = load_models()
    models['teller_scene2seq_a'].max_rounds = 20
    models['teller_scene2seq_aux2_a'].max_rounds = 20
    models['teller_rl_a'].max_rounds = 20
    # TODO(nikita): change max_rounds for partition-b tellers, too
    tellers = make_pairs(models,
        'teller_nn',
        'teller_pragmaticnn',
        'teller_scene2seq',
        'teller_scene2seq_aux2',
        'teller_rl',
    )

    drawers = make_pairs(models,
        'drawer_nn',
        'drawer_bowcanvas2bce',
        'drawer_lstmaddonly',
    )

    run_model_pairs(tellers, drawers)
