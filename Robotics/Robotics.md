
1. Robot Motion
	- define 2 coordinate frames: world frame W and robot frame R
	- degrees of freedom (DOF): 
		- rigid body that translates and rotates along 1D path - train - 1 DOF (1 translational)
		- rigid body that translates and rotates on 2D plane - ground robot - 3 DOF (2 translational, 1 rotational)
		- translates and rotates in 3D volume - flying robot - 6 DOF (3 trans, 3 rot.)
		- holonomic robot is one that can move instantaneously in any direction in the space of its degrees of freedom
	- holonomic ground robot - possible with omnidirectional wheels
	- standard wheel configurations - drive and steer (car), differential drive - both non holonomic - both use 2 motors each
	- differential drive 
		- two driving wheels on left and right sides of robot, each driven by its own motor. steering done by setting different wheel speeds. straight line motion if v_L = v_R. turns on spot if v_L = -v_R
		- ![[Pasted image 20230315213006.png]]
	- Car
		- two motors - one to drive, one to steer
		- follows circular path with fixed speed and steering angle.
		- four wheels - need rear differential and variable ('ackerman') linkage for steering
		- ![[Pasted image 20230315213728.png]] where R_d = L / sin(s)
	- robot speed v = r_w * omega (where omega (w) = angular velocity, r_w = radius of wheel)
	- DC motors:
		- power signal is sent to motor using PWM - pulse width modulation. we set the amount of power to be sent. Most often this is a voltage signal with a fixed amplitude but with the amount of 'fill-in' set using PWM
		- for precision, encoders and feedback can be used for servo control using a PID control law.
		- lego motor has encoder that records angular position. 
		- Principle: decide where we want the motor to be at every point in time. At a high rate, check where the motor actually is from the encoder. Record the difference (the error). Send a power demand to the motor depending on the error, aiming to reduce it. • Our motors: record motion rotational position in degrees. • Two main modes: position control (where demand is a constant) and velocity control (where demand increases linearly with time).
		- ![[Pasted image 20230316110956.png]]
		- ![[Pasted image 20230316111126.png]]
		- ![[Pasted image 20230316111212.png]]
		- position-based planning
			- turn to next waypoint and drive straight towards it.
			- ![[Pasted image 20230316111353.png]]
			- tan inverse can be achieved with atan2(dy, dx) in python
			- ![[Pasted image 20230316111502.png]]
		- local planning: dynamic window approach
			- robot wants to plan path around complicated set of obstacles.
			- Consider robot dynamics and possible changes in motion it can make within small time dt. 
			- For each possible motion look ahead longer time τ . Calculate benefit/cost based on distance from target and obstacles. 
			- Choose the best and execute for dt, then do it again.
			- ![[Pasted image 20230316111756.png]]
			- ![[Pasted image 20230316111909.png]]
		- Global planing: wavefront method
			- brute force 'flood fill' breadth first search of whole environment.
			- guaranteed to find shortest route, but slow
		- global planning: rapidly exploring randomised trees (RRT) method
			- Algorithm grows a tree of connected nodes by randomly sampling points and extending the tree a short step from the closest node. 
			- Expands rapidly into new areas, but without the same guarantees.


2. Sensors
	- sensors are propriocetive (self-sensing) or exteroceptive (outward-looking)
	- self sensing sensors (motor encoders, internal force sensors) improve robot's sense of internal state and motion
	- outward looking sensors needed for:
		- Localise without drift with respect to a map. • Recognise places and objects it has seen before. • Map out free space and avoid obstacles. • Interact with objects and people. • In general, be aware of its environment.
	- Proprioceptive sensors:
		- value of measurement z_p is a function of the state of robot x. z_p = z_p(x)
		- may depend on not just current state, but also previous states or current rate of change of state.
	- Outward-looking sensors:
		- measurement depends on state of robot x and state of world around it y: z_o = z_o(x, y)
	- single value sensors: return a single value within a given range (eg: touch, light, sonar sensors)
	- multiple value sensors: return array of values (eg: camera or laser range-finder)
	- touch sensors:
		- binary on/off state - no processing
		- switch open - no current flows
		- switch closed - current flows (hit)
	- light sensors:
		- detect intensity of passive light incident from forward direction with some range of angular sensitivity
		- multiple sensors in diff directions can guide steering
		- Lego sensors can also emit their own light which reflect off close targets and can be used to follow a line on the floor or for effective short-range obstacle avoidance
	- sonar (ultrasonic) sensors
		- measures depth (distance) by emitting ultrasonic pulse and timing interval until echo returns. sonar beam has agular width 10-20 degrees.
		- Fairly accurate depth measurement (centimetre) in one direction but can give noisy measurements in the presence of complicated shapes. Maximum range a few metres.
	- external sensing: laser range-finder
		- measures depth using active signal.
		- accurate (sub-millimetre). works on most types of surfaces.
		- commercial ladar sensors return array of depth mesurements from scanning beam
		- 2D and 3D scan verisons available
		- bulky and expensive for small robots
	- external sensing: vision
		- camera measures passive light intensity from multiple directive simultaneously by directing incident light onto light sensitive chip
		- returns large rectangular array of measurements
		- A single camera measures light intensity, rather than any direct information about geometry. 3D information processing and matching with data from either multiple cameras a single moving one.
	- ![[Pasted image 20230316115958.png]]
	- Strategies after collision:
		- try to move around obj: reverse, turn fixed angle and proceed
		- random bounce: rotate random angle and proceed until next collision
	- servoing
		- robot control technique where control params (like desired speed of motor) are coupled directly to sensor reading and updated regularly in negative feedback loop. (also called closed loop control)
		- needs high frequency update of sensor or motion may oscillate
		- control demand is set which aims to bring current value of the sensor to desired value
		- Proportional control: set demand proportional to negative error (difference between desired sensor value and actual sensor value): e.g. set velocity proportional to error: v = -k_p (z_desired - s_actual) where k_p is proportional gain const
		- ![[Pasted image 20230316121518.png]]
	- wall following with sonar:
		- use sideways looking sonar to measure distance z to wall.
		- use velocity control and a loop (at 20Hz for example)
		- With the goal of maintaining a desired distance d, set difference between left and right wheel velocities proportional to difference between z and d: v_R - v_L = K_p (z - d)
		- Symmetric behaviour can therefore be achieved using a constant offset v_C : 
			- v_R = v_C + (1/2) * K_p * (z - d)
			- v_L = v_C - (1/2) * K_p * (z - d)
	- Probabilistic sensor modelling
		- we can build a probabilistic measurement model for how it works based on the uncertainty of sensor measurements. This will be a probability distribution (specifically a likelihood function) of the form: p(z_o | x, y) - has Gaussian shape
		- ![[Pasted image 20230316122516.png]]
		- ![[Pasted image 20230316122546.png]]
		- Temporal filtering: e.g. smoothing or finding the median of the last few measurements of a sensor which reports a single reading such as a sonar. This is good at reducing the effect of occasional large outliers, which are measurements with an unusually large amount of error.
		- Geometric fitting (can be called feature detection) to data from a sensor which reports an array of measurements, such as a laser range finder or scanning sonar, where we might fit geometric shapes such as straight lines or corners to the measurements and output the parameters of those shapes rather then the raw measurements.



3. Probabilistic Motion
	- Scatter due to uncontrollable factors: variable wheel slip, rough surface, air currents etc
	- Systematic error: callibration etc.
	- zero mean errors: gaussian (i.e. normal) distribution
	- ![[Pasted image 20230316184557.png]]
	- Here e, f and g are zero mean noise (i.e. ‘uncertainty’) terms, with zero mean and a Gaussian distribution, which model how actual motion might deviate from the ideal trajectory.
	- ![[Pasted image 20230316191919.png]]
	- ![[Pasted image 20230316191951.png]]
	- ![[Pasted image 20230316192204.png]]
	- ![[Pasted image 20230316192343.png]]
	- Monte Carlo Localisation: 
		- cloud of particles represent uncertain robot state: more particles in a region = more probability that robot is there
		- Particle distribution:![[Pasted image 20230316203409.png]]
		- when spreading the particles after moving the robot, scale the variance proportional to the linear or angular distance moved. (eg: two 1m steps cause same spread as one 2m step)
		- when we achieve measurement z, we update particle weight as: 
		  w_i(new) = P(z | x_i) * w_i 
		  where P(z|w_i) is likelihood of particle i: probability of getting measurement z given that it represents the true state.
		- make point estimate of current position and orientation of robot by taking mean of all particles. x_mean = summation(i=1 to N) w_i * x_i



4. Monte Carlo Localisation
	- when wall following with sonar, if the angle between the robot's direction and  wall gets too large then it produces garbage reading
	- fix by mounting the sonar forward of the wheels. or use ring of sensors
	- In MCL, a cloud of weighted particles represents the uncertain position of a robot. Two ways of thinking about how MCL works: 
		- A Bayesian probabilistic filter. 
		- ‘Survival of the fittest’, like a genetic algorithm. After motion prediction, the particles are in a set of random positions which should span the possible positions into which the robot might have really moved. When a measurement is made from sonar, the likelihood each particle is assigned, according to how well it fits the measurement, is a ‘fitness’ score. Normalising and resampling then allow strong particles to reproduce (multiple copies are made of them), while weak particles die out.
	- Continuous localisation is a tracking problem: 
		- given a good estimate of where the robot was at the last time-step and some new measurements, estimate its new position.
		- assume we start from a perfectly known robot position - set state of all particles to same value (x_1 = x_2 = ... = x_init). set all weights to be equal (= 1 / N where N is no. particles)
	- Global localisation is often called the ‘kidnapped robot problem’: 
		- the environment is known, but the robot’s position within it is completely uncertain. 
		- state of each particle samples randomly from all possible positions within given region (x_i = Random). set weights to be equal (= 1/N). 
	- Steps in particle filtering:
			- Motion Prediction based on Proprioceptive Sensors. 
			- Measurement Update based on Outward-Looking Sensors. 
				- how to get the ground truth value
				- ![[Pasted image 20230316212633.png]]
				- ![[Pasted image 20230316212822.png]]
				- ![[Pasted image 20230316213104.png]]
				- ![[Pasted image 20230316213205.png]]
			- Normalisation. 
				- ![[Pasted image 20230316213232.png]]
			- Resampling.
				- To generate each of the N new particles, we copy the state of one of the previous set of particles with probability according to that particle’s weight. 
				- This is best achieved by generating the cumulative probability distribution of the particles, generating a uniformly distributed random number between 0 and 1 and then picking the particle whose cumulative probability this intersects.
				- ![[Pasted image 20230316213426.png]]



5. Advanced Sensing: Place Recognition and Occupancy Mapping
	- Kidnapped robot method solutions:
		1. simplest - initialize a large no. of particles randomly spread through environment and run filter normally. requires many particles and may take many movements and measurements to find its location
		2. could improve simplest method with more informative sensing - a ring of sonar sensors making measurements simultaneously
		3. one depth measurement and resampling - after a measurement (eg: sonar depth - 20cm) the weights of particles consistent with it will increase. movement and further measurements are needed to lock down position and ambiguities may still arise.
		4. using a compass and sonar together - ambiguity reduced with compass measurement eg: sonar depth = 20cm, compass bearing = 45 degrees.
		5. alternative relocalisation technique 
			- make a lot of measurements at a number of chosen locations and learn their characteristics. Can be done without prior map but needs training. Robot can only recognise locations it has learnt.
			- place robot in each target location to learn its appearance.
			- spin robot and take regularly spaced set of sonar measurements (eg: one per degree) - these are raw measurements stored to describe the place  (i.e. signature)
			- make histogram of raw measurements
			- when robot is placed in one of these locations, it takes the set of measurements to determine what position it is in: two histograms can be compared with a correlation test (measure sum of squared differences) and the saved location with lowest D_k (where D_k is difference bw histograms) is most likely candidate. If D_k is below a threshold then it is at known location, otherwise unknown location. Here H_m is new measurement histogram and H_k is known/saved signature histogram![[Pasted image 20230316214952.png]]
			- However, the robot may be rotated from the original histogram. instead of comparing every histogram shift, we build a signature invariant to robot rotation - such as a histogram of occurences of certain depth measurements (rather than depth measurement at every degree).
			- once correct location has been found, the shifting procedure to find the robot's orientation need only be carried out for that location.
	- Probabilistic Occupany Grid Mapping
		- robot's localisation is known. Which parts of the environment around a robot are navigable free-space, and which contain obstacles?
		- we use a grid representation instead of building parametric map of positions.
		- occupancy grids accumulate uncertain info from sensors like sonar to solidify towards precise maps.
		- We define an area on the ground we would like to map, and choose a square grid cell size. 
		- For each cell i, we store and update a probability of occupancy P(O_i) that it is occupied by an obstacle. 
		- P(E_i) is the corresponding probability that the cell is empty, where P(O_i) + P(E_i) = 1. 
		- We initialise the occupancy probabilities for unexplored space to a constant prior value; for instance 0.5 
		- Occupancy maps are often visualised with a greyscale value for each cell: from black for P(O_i) = 1 to white for P(O_i) = 0; intermediate values are shades of grey.
		- For each cell we want to update the probability of occupancy to take account of a new sonar measurement Z. 
		- Suppose that the sonar reports a depth Z = d. This provides evidence that cells around distance d in front of the robot are more likely to be occupied. But also, that cells in front of the robot at depths less than d are more likely to be empty. 
		- A sonar beam is not a perfect ray but has a width (e.g. 10–15◦ as it spreads out and we can take account of this as shown.
		- For each cell, we must test if it lies within the beam given the robot’s position. We do not learn anything about cells beyond the beam width or beyond the measured depth.
		- ![[Pasted image 20230316220913.png]]
		- ![[Pasted image 20230316220955.png]]
		- ![[Pasted image 20230316221255.png]]




6. Simultaneous Localisation and Mapping (SLAM)
	- 
