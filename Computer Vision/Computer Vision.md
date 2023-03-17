
1. Image Filtering
	- Moving average:
		- Image size: $N * N$
		- Kernel size: $K * K$
		- $N^2 * K^2$ multiplications and $N^2 * (K^2)$ summations
		- Complexity: $O(N^2 * K^2)$
	- Separable filter: uses convolution
		- Complexity: $O(N^2 * K)$
	- Types of filters:
		- Identity filter
		- Low-pass or smoothing filter: 
			- moving average
			- 2D gaussian:
				- $h(i, j) = (1 / (2 * pi * sigma^2)) * e^(-(i^2 + j^2)/(2 * sigma^2))$
				- i,j are offset from center of window
				- sigma controls shape of filter
				- k = 3 or 4
				- Separable: 
					- $h_x(i) = (1 / (sqrt(2*pi) * sigma)) * e^(-i^2/(2 * sigma^2))$
					- $h_y(j) = (1 / (sqrt(2*pi) * sigma)) * e^(-j^2/(2 * sigma^2))$
			- low-pass smoothen or blur image and keep low-frequency signals
		- high pass or sharpening filter
			- highlight the high frequency signals and sharpen image
			-  ![[Pasted image 20230311230423.png]]
		- denoising filter: 
			- median filter - replace the center pixel of the sliding window/kernel with the median of the window, non-linear filter
		 - Impulse response:
			 - For continuous signal, impulse response is Dirac delta function where $delta(x) = +$inf if x=0 OR 0 otherwise, such that integration of delta from -inf to +inf = 1
			 - For discrete signal, impulse response is Kornecker delta function delta[i] where $delta[i] =$ 1 if x=0 OR 0 otherwise.
			 - The output g[n] = f[n]  *  h[n] where * is convolution and h is impulse = summation(m=-inf to m=+inf) f[m] * h[n - m]
			    = summation(m=-inf to m=+inf) f[n - m] * h[m]
			- Note: convolution is commutative and associative
			- g[n, m] = f[n, m] * h[n, m] 
			   = summation(i=-inf to +inf) summation(j=-inf to +inf) 
			       f[i, j] * h[n - i, m - j]
			 - If a filter is a time-invariant system, then when we shift the input by time step 'k' then the output also shifts by time step 'k'.
				 - eg: g[n] = 10 * f[n] is time invariant and amplifies input by const.
				 - g[n] = n * f[n] is not time invariant as it depends on time step n.
			- If a filter is a linear system, combining t input singals linearly gives an output that is combined linearly. $output(a*f_1[n] + b*f_2[n]) = a*g_1[n] + b*g_2[n]$
			- ![[Pasted image 20230311234639.png]]
			- ![[Pasted image 20230311234735.png]]
			- ![[Pasted image 20230311234807.png]]




2. Edge Detection:
	- Prewitt Filter
		- ![[Pasted image 20230312003640.png]]
		- ![[Pasted image 20230312003802.png]]
	
	- Sobel filter
		- ![[Pasted image 20230312004146.png]]
		- ![[Pasted image 20230312004218.png]]
		- ![[Pasted image 20230312004533.png]]
	
	- Smoothing in edge detection
		- prewitt filters use [1,1,1] kernel for smoothing
		- sobel filters use [1,2,1] kernel for smoothing
	- Binary edge map
		- Canny edge detection
			- Perform Gaussian filtering to suppress noise: low sigma detects fine features, high sigma detects large scale edges and smooths out fine details.
			- Calculate the gradient magnitude and direction (prewitt or sobel). 
			- Apply non-maximum suppression (NMS) to get a single response for each edge. - puts pixel as M(x,y) if it is local maximum, else 0. perform image interpolation (nearest neighbour or linear interpolation) for pixels p and r if not located on pixel lattice when calculating if q is local max, or you can round the gradient direction into 8 angles (0, 45, 90...315) to avoid interpolation. if p and r are on pixel lattice, then find gradient qr and qp to determine if q is local max.
			- Perform hysteresis thresholding to find potential edges. 
				- binary thresholding: binary(x,y) = 1 if I(x,y) > t or 0 otherwise where t is threshold.
				- hysteresis threshold: two thresholds - t_high and t_low. If I(x,y) > t_high then edge, if I(x,y) < t_low then rejected, if t_low <= I(x,y) <= t_high, then check neighbouring pixels, if it is connected to an edge pixel then it is edge
		- Goals attained using Canny edge detection:
			- Good detection - Gaussian smoothing to suppress noise (reducing false positives), Hysteresis thresholding to find weak edges (reducing false negatives).
			- Good localisation - Use gradient orientation and non-maximum suppression to find the location of edges. 
			- Single response - Non-maximum suppression.
		 - Learning-based edge detection: convolutional neural network



3. Hough Transform
	- edges can be denoted as lines instead of a set of points, i.e. y = mx + b, OR x/a + y/b = 1 (where a and b are x and y intercepts), x*cos(t) + y*sin(t) = p (p is perpendicular dist from origin to line)
	- to find the parameter space from image space, we minimize the fitting error: min(m,b) summation(i) [y_i - (mx_i + b)]^2
	- Algorithm:
		- Initialise the bins H(p, theta) to all zeros. 
		- For each edge point (x, y) 
			- For theta from 0 to π:
				- Calculate p = xcos(theta) + ysin(theta) 
				- Accumulate H(p, theta) = H(p, theta) + 1 
		 - Find (p, theta) where H(p, theta) is a local maximum and larger than a threshold. The detected lines are given by p = xcos(theta) + ysin(theta)
	- Hough transform can detect multiple lines and is robust to noise and robust to object occlusion 
	- Circle detection: 
		- (x - a)^2 + (y - b)^2 = r^2. 
		- If r is unknown, then set a range for r = (r_min, r_max) and vote for a,b in H(a,b) for each r in the range, increasing by one pixel at a time.
		- x = a + rcos(theta), y = b + rcos(theta). gives idea for acceleration and we know the theta (i.e. its direction) as it comes from an edge map. so we only vote in theta direction
		- Circle detection using Hough transform:
			- Initialise the bins H(a,b,r) to all zeros. 
			- For each possible radius r in [r_min, r_max] 
				- For each edge point (x, y) 
					- Let theta to be gradient direction, or opposite gradient direction 
					- Calculate a = x - r*cos(theta), b = y - r*sin(theta)
					- Accumulate H(a,b,r) = H(a,b,r) + 1 
			- Find (a,b,r) where H(a,b,r) is a local maximum and larger than a threshold. The detected circles are given by x = a + rcos(theta), y = b + rsin(theta)
	- Hough transform has high complexity as each edge point needs a 2D or 3D vote for the parameter space, and we need to carefully set the params for edge detector, range of radius for circle, threshold for accumulator.
	- Other shapes:![[Pasted image 20230312043249.png]]
	- When we vote in the parameter space, we can add some weights. Instead of using equal vote for each edge point, the vote can be weighted by the gradient magnitude, so stronger edge points get higher weights.
	- Use random forest for voting for pedestrian walking detection in Hough space. predicts a displacement vector from the patch centre, given the image feature of the patch



4. Interest Point Detection
	- Harris Corner Detector
		- Take a small window of pixels and move them along an edge in edge map. if intensity shifts only in one direction then it is an edge. If it shifts in both directions then corner. 
		- Change in intentsity in window shift [u,v] is E(u,v) =  ![[Pasted image 20230312044541.png]]
		- ![[Pasted image 20230312044734.png]]
		- Harris detector is not scale variant, so we add another parameter to count in the scale (sigma). (we can get mutli scale images by gaussian smoothing with a different sigma, or sampling with diff spatial resolutions)![[Pasted image 20230313030130.png]]
		- Detector response R = lamda_1 * lamda_2  -  k * (lamda_1 + lamda_2)^2
			- Note: lamda1 >> lamda2 or lamda2 >> lamda1 means edge, lamda1 ~ lamda2 means corner, lamda1 and lamda2 = 0 means within edges. Hence when lamda1 * lamda2 is large, then represents corner. For each pixel, we perform gaussian smoothing with sigma, then find I_x and I_y, then find M, then find R. If R is above a certain threshold, and the interest point is local maxima across scale and space, then we consider that pixel to be a corner.
		- Detector response R = det(M) - k * (trace(M))^2. (trace is sum of diagnol elems)
		- I_x = G_x * I,  I_y = G_y * I    where G can be sobel filter, I is image, I_x and I_y are derivatives of image.
		- Regular harris detector is rotation invariant as you still get same change in intensity if an image is rotated
		- We can also use Laplcain Gaussian i.e. Gaussian smoothing followed by Laplacian. Laplacian: sum of second derivatives:![[Pasted image 20230313165650.png]]
		- second derivative is more sensitive to noise, which is why we first smooth the image before applying laplacian. Laplacian of gaussian is laplacian func f convoluted with:![[Pasted image 20230313170036.png]]
		- To factor in scale in LoG: ![[Pasted image 20230313170727.png]]
		- Difference of Gaussians (DoG) is a good approximation to normalized LoG and provides convenience in calculating response across diff scales.![[Pasted image 20230313170932.png]]
		- ![[Pasted image 20230313171156.png]]
		- ![[Pasted image 20230313171439.png]]



5. Feature Desciption
	- SIFT Algorithm:
		- single pixel - not indicative of local content (edge etc), sensitive to absolute intensity value
		- patch intensities - represent local pattern (sensitive to absolute intensity value and not rotation invariant)
		- Gradient orientation - robust to change of absolute intensity values (not rotation invariant)
		- Histogram - robust to rotation and scaling (sensitive to intensity change)
		- combine the advantages of gradient orientation and histogram to get SIFT (scale invariant feature transform)
		- SIFT steps:
			- detection of scale-space extrema (detection)
				- find interest points using gaussian difference for different scales sigma. eg: sigma, sqrt(2)*sigma, 2*sigma etc.
			- keypoint localization (detection)
				- for DoG defined as D(x) where x = (x, y, sigma)transpos:
				- we move from x to delta(x) to come to refined estimate![[Pasted image 20230313201712.png]]
			- orientation assignment (description)
				- divide the image into 16 subregions (4x4)
				- in each region calculate the gradient orientation of samply points
				- calculate histogram for each subregion with 8 bins (0, 45, 90, 135, 180, 225, 270, 315 degrees). for each bin count the sum of gradient magnitude for this orientation. 
				- This gives the SIFT feature descirptor.
				- dimension of descriptor: 16 subregions * 8 bins = 128
				- use a feature vector of 128 elems to describe each keypoint
			- keypoint descriptor (description)
			- to make descriptor invarient to scaling and rotation:
				- scale is known when keypoint is detected.
				- to know domain orientation:
				- gradient orientation of all pixels in neighbourhood of keypoint X vote for domain orientation:
					- orientation histogram with 36 bins from 0 degrees to 360 degrees
					- each pixel votes for an orientation bin. weighted by gradient magnitude
					- keypoint is assigned to an orientation corresponding to max of histogram
			- keypoint matching:
				- for each keypoint in image A, identify nearest neighbours in set of keypoints for image B. Distance defined by euclidean distance of SIFT descriptors.
				- we can find approximate nearest neighbours rather than exact ones (approx has faster algorithms)
				- for a keypoint (x, y) in image A (found after nearest neighbour search) we have a matching keypoint (u, v) in image B.![[Pasted image 20230313212026.png]]
				- ![[Pasted image 20230313212120.png]]
				- If some keypoints are mismatched (outliers), then we use Random Sample Consensus (RANSAC) to improve robustness in matching:
					- choose two random points/samples
					- fit a model (join the points with a line that passes through them)
					- based on the model/line, check how many other samples/points are inliers (i.e. is it the line of best fit?)
					- terminate after certain iterations or once enough inliers have been found
				- we can use keypoint matching to stitch images together if their keypoints match
				- we can use keypoint matching for object recognition
	- SIFT needs to calculate gradient magnitudes and orientation which can be slow in real time. We can implement it on FPGA (Field Programmable Gate Array) which is faster than CPU. SURF is faster than SIFT
	- SURF:
		- Speeded-Up Robust Features
		- Only calculates gradients along horizontal and vertical directions using Haar wavelets instead of histograms of gradient orientations in 8 bins
		- SURF applies very simple filters d_x and d_y onto sample points to calculate gradients along x and y. They are called Haar wavelets.
		- For each subregion, sum up the Haar wavelet responses over sample points (summing up pixel intensities with weights +1 or -1). The descriptor for this subregion is defined by 4 elems: summation(d_x), summation(d_y), summation(absoluteValue(d_x)), summation(absoluteValue(d_y))
		- if all elem values are low - homogenous region
		- sum(abs(d_x)) is high and others low - zebra pattern
		- sum(d_x) and sum(abs(d_x)) are high and others low - gradually increasing intensities.
		- Dimensions of SURF: 4x4 subregions. 4 elems per subregion = 64D vector (16 sub * 4 elems)
		- SURF is 5 times faster than SIFT
	- Binary Robust Independent Elementary Features (BRIEF):
		- Haar wavelets in SURF compare a local region to another and calculate the difference to get a floating point number.
		- BRIEF compares point p to q and gets binary val output. (tao(p, q) = 1 if I(p) < I(q), 0 otherwise)
		- randomly sample n_d pairs of points for binary tests. apply that same pattern to all interest points. if n_d = 256, then perform 256 tests which give us 1 bit each. hence BRIEF decriptor is n_d-simensional bitstring
		- if n_d = 256 then BRIEF descriptor is 256-bits = 32 bytes. SIFT is 512 bytes and SURF is 256 bytes. 
		- We only compare 2 numbers without calculating gradient orientation (SIFT) or intensity difference (SURF). Comparing 2 BRIEF descriptors is faster as we dont need to calculate Euclidean distance. 
		- example: perform 8 binary tests and get 1 byte of descriptor:
			- descriptor = ((I(p1) < I(q1)) << 7) + ((I(p2) < I(q2)) << 6) + ((I(p3) < I(q3)) << 5) + ((I(p4) < I(q4)) << 4) + ((I(p5) < I(q5)) << 3) + ((I(p6) < I(q6)) << 2) + ((I(p7) < I(q7)) << 1) + ((I(p8) < I(q8)) << 0) where << n is shifting the bit by n places.
		- Perform image matching and comparing 2 BRIEF descriptors using Hamming distance (i.e. using bitwise XOR operation followed by a bit count). XOR: if bits are different then 1, if same then 0. Then count the number of 1's in result
		- BRIEF does not account for rotation or scaling. It assumes images are taken from a moving camera that only involves translation.
		- BRIEF is 40 times faster than SURF and 200 times faster than SIFT
	- Histograms of Oriented Gradient (HOG)
		- Similar to SIFT. Uses gradient orientation histograms.
		- HOG decribes features for a large image region rather than just around a point (SIFT)
		- Divides large region into a dense grid of cells, describes each cell, concatenates the local descriptions to form global description.
		- Divide image into equally spaced cells. (ex: cell contains 8x8 pixels). 4 cells form a block. describe block (ex: top left corner)
		- move to next block. use orientation histogram to describe content. there is overlap between blocks.
		- for each block, descriptor vector v (concat of 4 histograms) is normalised:![[Pasted image 20230314140128.png]]
		- concat normalised local descriptors for all blocks to describe the full image/large image region.
		- this can be used for image classification based on image features, detect if image region contains human or not, retrieve similar images by their features.

 
 
 
 6. Image classification: 
	 - ![[Pasted image 20230314140533.png]]
	- divide dataset into training set and test set
	- MNIST dataset: handwritten image recognition dataset (60,000 training samples, 10,000 test samples) - each sample is a digit bw 0-9
	- pre-processing:
		- normalize size of each image to 28x28
		- normalise location, place centre of mass of digit to centre of image
		- perform slant correction (make principal axis vertical by shifting each row in img)
	- feature extration is carried out by either hand-crafted algorithms (like pixel intensities, HOG etc) or using learnt features (eg: CNN)
	- Classifier:
		- K nearest neighbours
			- non-parametric classifier
			- if k=1 then each test data point is assigned the class of its nearest neighbour
			- k>1 then compute k nearest neighbours and test data point is assigned class by majority voting
			- ![[Pasted image 20230314142148.png]]
			- ![[Pasted image 20230314142208.png]]
			- use euclidean distance to compare feature vectors
			- k is hyperparameter.
			- hyperparamter tuning: cross-validation
				- divide training data into folds (ex: 5)
				- 4 folds as training, 1 as validation set
				- cycle through each fold and train model 5 times, calculating the average performance for some hyper-parameter values.
				- once hyper-parameter values are chosen, train the model on full training set using that hyper-parameter. then test on test set.
			- advantages of knn:
				- no training step
				- simple but effective
				- multi-class classification
			- disadvantages:
				- storage and search expensive. all training data need to be stored and searched
			- complexity = O(MN) for N training images and M test images. No training time. Slow test time
			- euclidean distance not invariant to scale, rotation etc. if used bw full images.
		- Support vector Machine
			- Linear SVM is line that separates two different classes.
			- ![[Pasted image 20230314143321.png]]
			- KNN needs to store all training data but linear classifier can discard training data after knowing w and b.
			- ![[Pasted image 20230314143419.png]]
			- To determine maximum margin hyperplane, we need innermost points, called support vectors.
			- ![[Pasted image 20230314144103.png]]
			- ![[Pasted image 20230314144159.png]]
			- ![[Pasted image 20230314144220.png]]
			- ![[Pasted image 20230314144245.png]]
			- ![[Pasted image 20230314144909.png]]
			- ![[Pasted image 20230314144937.png]]
			- ![[Pasted image 20230314145014.png]]
			- Classifier: w.x + b = 0
			- Use HOG descriptor as x
			- The loss function above needs to be optimized since the summation to N, N is over a million. So we use Stochastic gradient descent (SGD):
				- ![[Pasted image 20230314145449.png]]
			- at test time, we perform classification: c = +1 if wx+b >= 0, -1 otherwise
			- For multi-class classification:
				- ![[Pasted image 20230314145914.png]]
				- ![[Pasted image 20230314145929.png]]
		- neural network
			- Take simplest case: 1 neuron. takes input, applies activation function and generates output.![[Pasted image 20230314150553.png]]
			- Commonly used activation function is sigmoid function (i.e. logistic function)![[Pasted image 20230314150608.png]]
			- ![[Pasted image 20230314150807.png]]
			- ![[Pasted image 20230314150845.png]]
			- Perceptron
				- only consists of single layer and uses Heaviside step function as activation function.
				- y = 1 if wx + b > 0, 0 otherwise
				- optimise w and b so than y matches ground truth.
			- Multi-Layer Perceptron (MLP)
				- put several layers of neurons into connection, where outpout of one neuron can be input into another.
			- Convolutional Neural Networks (CNN)
				- similar to MLP. CNN assumes that inputs are images and encode certain properties (ex: local connectivity, weight sharing) into the architecture. This makes computation more efficient and substantially reduces the number of params.
				- each neuron only sees small local region in the layer before it. the region it sees is called receptive field. local connectivity reduces params
				- 
			- Loss function:
				- Mean squared error:![[Pasted image 20230314234943.png]]
				- works for regression problems but may not work for image classification where y is categorical (as opposed to continuous variable). 
				- Instead we use binary classification(for 2 classes): output layer has 1 neuron, use sigmoid activation for last layer. the output value is between 0 and 1 (range of probability). Loss func is distance between predicted probability and true probability.
					- information theory provides us with a metric for probability distributions called the cross entropy. Cross entropy between true prob p and estimated prob q is:![[Pasted image 20230314235704.png]]
				- Multi-class classification:
					- For K classes, put K neurons as output layer.
					- use softmax func to make output of K neurons form probability vector:![[Pasted image 20230314235926.png]]
					- cross-entropy loss:![[Pasted image 20230315000027.png]]
					- where estimated probability q = [f(z_1), f(z_2),....,f(z_K)] and true probability p = [y_1....y_K]
				- Data augmentation improves classification performance. Affine transformation = translation, scaling, squeezing, shearing
				- MLP uses many parameters. May not scale up to bigger images. This is because a 2D image is considered a flattened vector without considering its 2D nature.
				- Forward propogation:![[Pasted image 20230315002516.png]]
				- gradient descent to minimise loss function J(W,b)![[Pasted image 20230315004109.png]]
				- gradient 
		- vision transformer