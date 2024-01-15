# DRFS: A Depth-Aware Method for Large-View Underwater Image Reconstruction and Fusion

## Code

###  Usage
	
	(1). We provide scripts that make it easier to test data. The following are the steps:
	(2). Download code and comile.
		You need to install the dependencies in require.txt.
	(3). Download dataset to "Input" folder.
	(4). Run project.
	
You can find results in folder "Output".

  




## Abstract
In the field of information fusion, consistency of geometric features is a key challenge in the task of perspective reconstruction. Existing underwater reconstruction methods, limited by complex underwater environments, cannot effectively use edge linear features. To overcome these challenges, we propose a new approach called DRFS. The DRFS method innovatively exploits the continuous edge curve features of objects in the underwater scene. It effectively maintains the large-scale structure of the image achieving efficient reconstruction of underwater images with large viewing angles and solving the problem of restricted underwater scenes. The proposed method cascades multiple procedures such as D-procedure, R-procedure, F-procedure, and S-procedure to reconstruct natural-definition large-view underwater images. The D-procedure combines prior knowledge and unsupervised methods to estimate the image depth information to provide it to R and F. The R-procedure recovers the image by means of a complex underwater imaging model, using the differences between atmospheric and underwater imaging and effectively solving the fog effect of the image, thus enhancing the contrast of the image. The F-procedure extracts point features based on depth information and constructs a large-scale structure with the edge curves of the underwater scene as main body. Hence, we solve the problem of the lack of available edge straight lines in the underwater scene and the inability to maintain edge consistency. The S-procedure uses global similarity and large-scale structure to achieve consistent geometric structures when registering multiple restored underwater images. It eliminates uneven transitions and brightness differences between overlapping areas through Gaussian pyramid fusion. The experimental results demonstrate that the proposed DRFS method can obtain reconstructed large-view underwater images that are more visually appealing and naturally visible.
