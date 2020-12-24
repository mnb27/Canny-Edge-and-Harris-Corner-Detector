# Canny-Edge-and-Harris-Corner-Detector
Understanding and developing any vision related model requires feature extraction from the
image for analysis. Edges and Corners are the features that are trivially identified by our
human eye. Thus extracting edges and images is of prime importance in computer vision.

The **Canny Edge Detector** is an edge detection technique that uses a multi-stage algorithm
flow. It was developed by John F. Canny in 1986. It extracts useful structural information
from different vision objects and dramatically reduce the amount of data to be processed. It
has been widely applied in various computer vision and imaging applications. Canny Edge
detector works on the principle that intensity changes suddenly across the edge, however
remains uniform along the edge.

![Bycycle](/data/bicycle.bmp)
![Bycycle Edges](/Results/canny_edge/bicycle/bicycle_final_edges.jpg)

**Harris Corner Detector** is a corner detection operator that is commonly used tool in computer
vision algorithms to extract corners and infer features of an image.
A corner is a point whose local neighbourhood stands in two dominant and different edge
directions. In other words, a corner is point of intersection of edges. Thus, while an edge
has a gradient change in any one direction, a corner has a gradients change along multiple
directions. The gradient of the corner (in both directions) have a high variation, which can
be used to detect it. This is the principle used in Harris corner detector.
Also, it is popular because it is rotation, scale and illumination invariant. It was first
introduced by Chris Harris and Mike Stephens in 1988.

![bird](/data/bird.bmp)
![bird](/Results/harris_corner/bird/_corner_points_marked.jpg)
