## GPT Custom

Allows ChatGPT to learn custom data from your PDFs.

### Prereq
- OPENAI key
- python3
- pip3

### Setup
```
pip3 install -r requirements.txt
```

### Example


`OPENAI_API_KEY=redacted PDF_FILE=robot-racing-edward.pdf make run`

Finished reading robot-racing-edward.pdf. Now you can ask me questions!

`Question (type exit to quit): So what's this paper about?`

 This paper is about the design and implementation of Blizzard, an autonomous race car developed by UBC Thunderbird Robotics' Snowbots Robot Racing Team for the 2011 Robot Racing Challenge.

`Question (type exit to quit): Who are the authors of this paper?`

 Navid Fattahi, Edward Han, and Eduardo Silva.

`Question (type exit to quit): What programming language did they use to create this autonomous race car?`

 They used C++ for their software framework, Robot Operating System (ROS).

`Question (type exit to quit): What was their navigation algorithm?`

 The navigation algorithm was split into two sections: Cone Detection, and Path Planning. Cone Detection uses an array of distances received from the LIDAR to generate a vector of cones with two properties: distance and angle. This vector is then passed to Path Planning, which uses the vector of cones to identify empty regions.

`Question (type exit to quit): How did the robot detect cones using LIDAR?`

 The LIDAR has a viewing range of -90 to 90 degrees, with a angle increment of 0.36 degrees. Using the array of distances received from the LIDAR, distances and their corresponding angles can be found. The Cone Detection algorithm (Figure 4) then uses this information to generate a vector of cones, each with two properties: distance and angle.

`Question (type exit to quit): How much did the robot cost?`

 The robot cost $3530.
