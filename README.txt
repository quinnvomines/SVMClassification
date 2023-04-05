Quinn Vo, CSCI 573

DESCRIPTION: The goal of this project is to create a model that is able to distinguish six different human behaviors using a support vector machine. Okease see FinalReport.pdf for more details.

IMPORTANT: 
MAKE SURE TO USE PYTHON 3. 
You MIGHT have to do an installation of LIBSVM with PYTHON 3. 
If you run the program and it says "No module named libsvm.svmutil", you must use Python 3. 

Install LIBSVM using the command "pip install -U libsvm-official"

This project does not use ROS. To compile code:
    - Run "python3 svm-classify.py RAD" to classify using RAD representations
    - Run "python3 svm-classify.py HJPD" to classify using HJPD representations
    - Run "python3 svm-classify.py HOD" to classify using HOD representations



RAD implementation: 

    - Accuracy achieved: 62.5%

    - Bins = 18, C = 0.03125, gamma = 0.5

    - Joints used: Joint 1 was used as reference joint. Joints 4, 8, 12, 16, 20 were used to calculate 
      relative distances and angles

    - How the histograms were computed: A histogram was made for each distance from joint
      to reference joint. A histogram was also made for each angle for a pair of joints, 
      using joint 1 as the reference joint. (5 for distances and 5 for angles). Outliers were removed; 
      histograms are normalized; histograms are concatenated. 10 histograms made per instance.

HJPD implementation: 

    - Accuracy achieved: 66.67% 

    - Bins = 18, C = 2.0, gamma = 0.5

    - Joints used: Joint 1 was used as reference joint. Joints 2-20 were used to calculate 
      relative distances.

    - How the histograms were computed: The same as RAD implementation, but
      HJPD implementation ignores pairwise angles. It is also different from the RAD
      implementation in that it calculates the distances using all the joints
      as opposed to selecting a handful. This means that 19 histograms 
      were made per instance. Outliers were removed; histograms are normalized;
      histograms are concatenated. 

HOD implementation: 

    - Accuracy achieved: 50.0%

    - Bins = 12, C = 128.0, gamma = 32.0

    - Joints used: All joints were used

    - How the histograms were computed: Each joint was projected into 2D cartesian plans. 
      Therefore, each joint had 3 projections (XY, XZ, YZ). For each pair of points in each projection, 
      the angle was calculated and placed in one of the 8 bins. The distance's magitude was used as the 
      histogram's real-value count. Histograms were divided by number of data points it contained. 
      Finally, they were concatenated. 
