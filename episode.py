# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    from IPython.display import display
except ImportError:
    assert not INTERACTIVE
    def display(*args, **kwargs):
        pass

import functools
from pathlib import Path
import datetime

import abs_render
import codraw_data
from abs_metric import scene_similarity

class Episode(list):
    def get_last(self, event_type):
        for event in reversed(self):
            if isinstance(event, event_type):
                return event
        return None

    def reconstruct(self):
        reconstructed_scene = []
        for event in self:
            if isinstance(event, codraw_data.DrawClipart):
                reconstructed_scene = [c for c in reconstructed_scene if c.idx != event.clipart.idx]
                reconstructed_scene.append(event.clipart)
            elif isinstance(event, codraw_data.DrawGroup):
                reconstructed_scene = [c for c in reconstructed_scene if c.idx not in [c2.idx for c2 in event.cliparts]]
                reconstructed_scene.extend(event.cliparts)
        return codraw_data.AbstractScene(reconstructed_scene)

    def display(self):
        scene = None
        for event in self:
            if isinstance(event, codraw_data.ObserveTruth):
                assert scene is None, "Multiple ObserveTruth events not allowed in an episode"
                scene = event.scene
            elif isinstance(event, codraw_data.SelectClipart):
                display(event.clipart)
            elif isinstance(event, codraw_data.DrawClipart):
                abs_render.display_cliparts([event.clipart], color='red', scale=0.75)
            elif isinstance(event, codraw_data.DrawGroup):
                abs_render.display_cliparts(event.cliparts, color='red', scale=0.75)
            elif isinstance(event, codraw_data.TellGroup):
                print("TELLER:", event.msg)
            elif isinstance(event, codraw_data.ReplyGroup):
                print("DRAWER:", event.msg)
            elif isinstance(event, codraw_data.TellerIntention):
                if event.drawn is not None:
                    abs_render.display_cliparts(event.drawn, color='purple', label='drawn', scale=0.33)
                if event.draw_next is not None:
                    abs_render.display_cliparts(event.draw_next, color='yellow', label='draw next', scale=0.33)
                if event.undrawn is not None:
                    abs_render.display_cliparts(event.undrawn, color='cyan', label='undrawn', scale=0.33)
        print('===')
        reconstructed_scene = self.reconstruct()
        abs_render.display_cliparts(scene, label='ground truth', scale=0.75)
        abs_render.display_cliparts(reconstructed_scene, color='red', label='reconstructed', scale=0.75)
        print('Similarity =', scene_similarity(reconstructed_scene, scene))

    def to_html(self):
        res = ""
        scene = None
        delayed_selected_clipart = ""
        for event in self:
            if isinstance(event, codraw_data.ObserveTruth):
                assert scene is None, "Multiple ObserveTruth events not allowed in an episode"
                scene = event.scene
            elif isinstance(event, codraw_data.SelectClipart):
                delayed_selected_clipart += abs_render.svg_from_cliparts([event.clipart], inline_images=False)
            elif isinstance(event, codraw_data.DrawClipart):
                res += delayed_selected_clipart
                delayed_selected_clipart = ""
                res += abs_render.svg_from_cliparts([event.clipart], color='red', inline_images=False)
            elif isinstance(event, codraw_data.DrawGroup):
                res += delayed_selected_clipart
                delayed_selected_clipart = ""
                res += abs_render.svg_from_cliparts(event.cliparts, color='red', inline_images=False)
            elif isinstance(event, codraw_data.TellGroup):
                res += f"<p>TELLER: {event.msg}</p>"
            elif isinstance(event, codraw_data.ReplyGroup):
                res += f"<p>DRAWER: {event.msg}</p>"
            elif isinstance(event, codraw_data.TellerIntention):
                if event.drawn is not None:
                    res += abs_render.svg_from_cliparts(event.drawn, color='purple', label='drawn', scale=0.33)
                if event.draw_next is not None:
                    res += abs_render.svg_from_cliparts(event.draw_next, color='yellow', label='draw next', scale=0.33)
                if event.undrawn is not None:
                    res += abs_render.svg_from_cliparts(event.undrawn, color='cyan', label='undrawn', scale=0.33)

        res += f"<p>===</p>"
        reconstructed_scene = self.reconstruct()
        res += abs_render.svg_from_cliparts(scene, label='ground truth', inline_images=False)
        res += abs_render.svg_from_cliparts(reconstructed_scene, color='red', label='reconstructed', inline_images=False)
        res += f"<p>Similarity = {scene_similarity(reconstructed_scene, scene)}</p>"
        return res

    def write_html(self, name_or_path):
        if isinstance(name_or_path, Path):
            path = name_or_path
        else:
            path = Path(f"./renders/{name_or_path}.html").resolve()
        assert not path.exists(), "File already exists!"
        assert path.parent.exists(), "Parent directory does not exist"
        path.write_text(self.to_html())

    def get_true_scene(self):
        scene = None
        for event in self:
            if isinstance(event, codraw_data.ObserveTruth):
                assert scene is None, "Multiple ObserveTruth events not allowed in an episode"
                scene = event.scene
        assert scene is not None, "Episode has no ObserveTruth events"
        return scene

    def scene_similarity(self):
        return scene_similarity(self.reconstruct(), self.get_true_scene())

    @classmethod
    def run(cls, scene, fns):
        episode = cls([codraw_data.ObserveTruth(scene)])
        while True:
            for fn in fns:
                if type(episode[-1]) in fn._trigger_types:
                    old_len = len(episode)
                    fn(episode)
                    if len(episode) == old_len:
                        return episode
                    break
            else:
                assert False, f"No response for event: {type(episode[-1]).__name__}"

    @classmethod
    def run_script(cls, scene_and_script, fns):
        scene, script = scene_and_script
        episode = cls([codraw_data.ObserveTruth(scene)])
        episode.script = script
        episode.script_index = 0

        while True:
            for fn in fns:
                if type(episode[-1]) in fn._trigger_types:
                    old_len = len(episode)
                    fn(episode)
                    if len(episode) == old_len:
                        return episode
                    break
            else:
                assert False, f"No response for event: {type(episode[-1]).__name__}"


def respond_to(*event_types):
    types = set([(x if issubclass(x, codraw_data.Event) else None) for x in event_types])
    assert None not in types, "Invalid event type in decorator"

    def deco(fn):
        if hasattr(fn, '_trigger_types'):
            fn._trigger_types |= types
        else:
            fn._trigger_types = types
        return fn
    return deco

def response_partial(fn, *args, **kwargs):
    res = functools.partial(fn, *args, **kwargs)
    res._trigger_types = fn._trigger_types
    return res

class Transcriber:
    def __init__(self, filename, scenes=None, scenes_description="", scenes_and_scripts=None):
        self.filename = filename
        if scenes is not None:
            self.scene_data = scenes
            self.use_script = False
        else:
            self.scene_data = scenes_and_scripts
            self.use_script = True

        self.scenes_description = scenes_description

    def __call__(self, name_or_path, description="", **partition_to_fns):
        if isinstance(name_or_path, Path):
            path = name_or_path
        else:
            path = Path(f"./renders/{name_or_path}.html").resolve()
        assert not path.exists(), "File already exists!"
        assert path.parent.exists(), "Parent directory does not exist"

        assert isinstance(description, str)

        res = ""
        res += f"<p>Filename: {self.filename}</p>"
        res += f"<p>Scenes: {self.scenes_description}</p>"
        res += f"<p>Started: {datetime.datetime.now()}</p>"
        res += f"<p>Description: {description}</p>"
        for partition, fns in partition_to_fns.items():
            res += f"<p></p>"
            res += f"<h2>Partition {partition}</h2>"
            for i, scene_datum in enumerate(self.scene_data):
                res += f'<h3 id="{partition}_{i}">Scene {i} <a href="#{partition}_{i}">[here]</a></h3>'
                if not self.use_script:
                    res += Episode.run(scene_datum, fns).to_html()
                else:
                    res += Episode.run_script(scene_datum, fns).to_html()
        path.write_text(res)
