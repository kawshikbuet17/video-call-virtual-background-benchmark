# Samples

Test images and videos used when no webcam is available (and inside Docker).

## Included

| File | Source | License |
|------|--------|---------|
| `person_portrait.jpg` | NASA portrait of Kalpana Chawla, 960x1200 copy from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kalpana_Chawla,_NASA_photo_portrait_in_orange_suit.jpg) | Public domain (NASA) |

No sample video is included yet (TODO: find a short, freely licensed one).

## Add your own

1. Copy an image (`.jpg`, `.png`) or video (`.mp4`) into this folder.
2. Run a model with it, for example:
   `python run.py --image ../assets/samples/my_photo.jpg`
   `python run.py --video ../assets/samples/my_clip.mp4`

Tips:
- One person, facing the camera, upper body visible, works best.
- To record your own clip on Windows, use the Camera app; it saves `.mp4` files.
- Keep files small (a few MB). Do not commit private photos or videos.
