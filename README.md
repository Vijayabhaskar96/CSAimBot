# CSAimbot - Aim Bot for FPS Games
## Using Tensorflow Object Detection API
##### Tested on Counter Strike 1.6 (Should work on most FPS games)

Inspired from Sentdex's Work on his "Python Plays GTA V" series
Thanks to,
1) Sentdex (https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)

2) Daniel Kukiela (https://twitter.com/daniel_kukiela) for keys.py file

3) Tensorflow Object Detection API

## Instructions
1) Proceed this tutorial on the below link and get Tensorflow Object Detection API Working.

    (https://goo.gl/ricx6n)

2) Clone this repository and extract in your desired location.

3) Copy paste the object_detection folder from the Step 1 into the CSAimBot cloned folder.

4) Get the game running in windowed mode on top left corner of the screen

   Use Borderless gaming to remove the title bar of the game window (not necessary,but recommended)
   (http://westechsolutions.net/sites/WindowedBorderlessGaming/download)

5) Command Line Arguments available:
```
    --help  : Displays all the available arguments and their usage.
    --width : Width of the game resolution(default:800)
    --height : Height of the game resolution(default:600)
    --resize : Keep this as low as possible to get better detection of person but decreasing it also reduces the frame rate of what                    bot sees. (default:4)
    --score : Increase as long as the bot detects the person,Decrease if bot can't detect the person.(default:0.40)
    --show : Set to False if you don't want to see the captured screen(default:True)
    --input : (Enter Without Quotes)Choose between "keyboard" and "mouse".(default:keyboard).Choose the --key if chose keyboard)
    --key : (Enter Without Quotes)Choose Anyone from 
    --shoot : (Enter Without Quotes) Shoots at CENTER of the person detected by default(choose between:CENTER,HEAD,NECK)
    --duration : How long to shoot(in seconds),default:0.4 seconds
```
    
6) Run the Run_Me.py file (Make sure CSAimBot directory is the current working directory,if not python will throw error)
    ```
    python Run_Me.py
    ```
    if you want use custom settings from Step 5,Example usage:
    ```
    python Run_Me.py --width 1024 -- height 768 --resize 3 --score 0.50 --show False --input mouse --shoot HEAD --duration 0.2 
    ```
