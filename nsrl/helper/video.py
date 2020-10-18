import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def frames_to_video(frames, path, scale=2.0):
    """
    convert a set of frames to a video, and save it to path
    :param frames: frames to save video to. time_steps x frame_dim
    :param path: path to save video to.
    :return: True if saved
    """

    width, height = frames.shape[-2:]
    width = int(width * scale)
    height = int(height * scale)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 30
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        frame = cv2.merge([frame, frame, frame])
        if scale != 1:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        out.write(frame.astype(np.uint8))

    out.release()

# Taken from https://stackoverflow.com/questions/45947608/rendering-a-unicode-ascii-character-to-a-numpy-array
def text_phantom(text, size=16):
    # Availability is platform dependent
    # font = 'arial'
    font = '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'

    # Create font
    pil_font = ImageFont.truetype(font, size=size,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [text_width, text_height], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = (0, 0)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    return np.average(255 - np.asarray(canvas), axis=-1)

def add_label_to_frames(frames, label, size=10):
    """
    Adds a label on the top left of frame
    :param frames: frames to add label (timesteps x width x height)
    :param label: label
    :param size: text size of label (in px)
    :return: frame with index on top left of dimension size.
    """
    label_image = text_phantom(str(label), size)
    lwidth, lheight = label_image.shape

    for i in range(len(frames)):
        frames[i, :lwidth, :lheight] = label_image

    return frames

if __name__ == "__main__":
    from examples.ALE.ALE_env_gym import MyEnv as ALE_env
    import os

    rng = np.random.RandomState()
    game = "MontezumaRevenge-v0"

    env = ALE_env(rng, game=game, timesteps_per_action=4, frame_size=(128, 128))

    env.reset(0)
    actions = list(range(env.nActions()))
    states = []

    for i in range(100):
        action = np.random.choice(actions)
        env.act(action.item())
        if env.inTerminalState():
            env.reset(0)
        frame = add_label_to_frames(env.state, i)
        states.append(frame)

    all_obs = np.stack(states)
    all_obs_flattened = np.concatenate(all_obs, axis=0)

    path = os.path.join(os.path.expanduser('~'), 'Documents', 'temp', 'test.mp4')

    frames_to_video(all_obs_flattened, path, scale=3.0)
    print("here")
