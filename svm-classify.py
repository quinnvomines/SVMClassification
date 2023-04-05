import math
import numpy as np
import os
import sys
from libsvm.svmutil import *


#Calculate point of interest with respect to reference point
def calc_point_interest(point, ref_point):
    np_point = np.array(point)
    np_ref_point = np.array(ref_point)
    return list(np_point - np_ref_point)

#Calculate magnitude
def calc_magnitude(point):
    sum = 0
    for i in range(len(point)):
        sum += point[i]**2
    return math.sqrt(sum)

#Calculate angles between two points
def calc_angle(point, point2):
    a = math.acos(np.dot(point,point2) / (calc_magnitude(point) * calc_magnitude(point2)))
    return a

#Remove outliers given numpy array
def remove_outliers(array):
    Q1 = np.percentile(array, 25)
    Q3 = np.percentile(array, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return array[np.where((array >= lower_bound) & (array <= upper_bound))]

def process_file_rad(directory, files, output_file):
    #Histogram hyperparameters
    dis_bins = 18
    ang_bins = 18
    lower_bound_dis = -0.5
    upper_bound_dis = 1.5
    lower_bound_ang = -1.0
    upper_bound_ang = 4.0

    #For each instance in train or test
    for file_index in range(len(files)):

        #Open file
        file_path = directory + files[file_index]
        f = open(file_path, 'r')
        
        #Read each line in file and store the frames and joints we need
        frame_joint_map = {}
        for read_line in f:
            #Convert line to list of floats
            processed_rl = [value for value in read_line[:-1].split(' ') if value != '']
            line = [float(i) for i in processed_rl] 

            #If not in map, add it
            if(line[0] not in frame_joint_map): 
                frame_joint_map[line[0]] = []
            
            #Add only select joints
            if(line[1] == 1 or line[1] == 4 or line[1] == 8 or line[1] == 12 or line[1] == 16 or line[1] == 20):
                frame_joint_map[line[0]].append([line[2],line[3],line[4]])

        #Go through each frame and check if nan values exist; if so, delete
        delete_frames = []
        for frame in sorted(frame_joint_map):
            for j in frame_joint_map[frame]:
                if(True in np.isnan(j)):
                    delete_frames.append(frame)
        delete_frames = set(delete_frames)
        for del_frame in delete_frames:
            del frame_joint_map[del_frame]
        
        #Close file
        f.close()
        
        #For each frame, compute and store distances and angles
        distances = []
        angles = []
        for frame in sorted(frame_joint_map):
            ref_point = frame_joint_map[frame][0]
            frame_distances = []
            frame_angles = []

            #Go through each joint and calculate distance with reference to body center
            for i in range(1,6):
                temp_val = (frame_joint_map[frame][i][0] - ref_point[0])**2
                temp_val2 = (frame_joint_map[frame][i][1] - ref_point[1])**2
                temp_val3 = (frame_joint_map[frame][i][2] - ref_point[2])**2
                d = math.sqrt(temp_val + temp_val2 + temp_val3)
                frame_distances.append(d)
            distances.append(frame_distances)
            
            #Calculate angles with each joint pair
            joint_pairs = [(1,3),(3,5),(4,5),(2,4),(1,2)]
            for i in range(len(joint_pairs)):
                p1 = calc_point_interest(frame_joint_map[frame][joint_pairs[i][0]], ref_point)
                p2 = calc_point_interest(frame_joint_map[frame][joint_pairs[i][1]], ref_point)
                a = calc_angle(p1,p2)
                frame_angles.append(a)
            angles.append(frame_angles)

        #Remove distance outliers
        d1 = remove_outliers(np.array(distances)[:,0])
        d2 = remove_outliers(np.array(distances)[:,1])
        d3 = remove_outliers(np.array(distances)[:,2])
        d4 = remove_outliers(np.array(distances)[:,3])
        d5 = remove_outliers(np.array(distances)[:,4])

        #Remove angle outliers
        a1 = remove_outliers(np.array(angles)[:,0])
        a2 = remove_outliers(np.array(angles)[:,1])
        a3 = remove_outliers(np.array(angles)[:,2])
        a4 = remove_outliers(np.array(angles)[:,3])
        a5 = remove_outliers(np.array(angles)[:,4])

        #Create distance histograms and normalize
        hist_d1 = np.histogram(d1, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d1))
        hist_d2 = np.histogram(d2, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d2))
        hist_d3 = np.histogram(d3, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d3))
        hist_d4 = np.histogram(d4, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d4))
        hist_d5 = np.histogram(d5, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d5))

        #Create angle histograms and normalize
        hist_a1 = np.histogram(a1, ang_bins, range=(lower_bound_ang,upper_bound_ang))[0] / float(len(a1))
        hist_a2 = np.histogram(a2, ang_bins, range=(lower_bound_ang,upper_bound_ang))[0] / float(len(a2))
        hist_a3 = np.histogram(a3, ang_bins, range=(lower_bound_ang,upper_bound_ang))[0] / float(len(a3))
        hist_a4 = np.histogram(a4, ang_bins, range=(lower_bound_ang,upper_bound_ang))[0] / float(len(a4))
        hist_a5 = np.histogram(a5, ang_bins, range=(lower_bound_ang,upper_bound_ang))[0] / float(len(a5))

        #Concatenate histograms
        output = np.concatenate((hist_d1,hist_d2,hist_d3,hist_d4,hist_d5,hist_a1,hist_a2,hist_a3,hist_a4,hist_a5))

        #Print to file
        f = open(output_file,'a')

        if(files[file_index][0:3] == 'a08'):
            f.write('8 ')
        elif(files[file_index][0:3] == 'a10'):
            f.write('10 ')
        elif(files[file_index][0:3] == 'a12'):
            f.write('12 ')
        elif(files[file_index][0:3] == 'a13'):
            f.write('13 ')
        elif(files[file_index][0:3] == 'a15'):
            f.write('15 ')
        elif(files[file_index][0:3] == 'a16'):
            f.write('16 ')

        for i in range(len(output)):
            if(i == len(output)-1):
                f.write(str(i+1) + ':' + str(output[i]) + '\n')
            else: 
                f.write(str(i+1) + ':' + str(output[i]) + ' ')
        f.close()

def process_file_hjpd(directory, files, output_file):
    #Histogram hyperparameters
    dis_bins = 18
    lower_bound_dis = -0.5
    upper_bound_dis = 1.5

    #For each instance in train or test
    for file_index in range(len(files)):

        #Open file
        file_path = directory + files[file_index]
        f = open(file_path, 'r')
        
        #Read each line in file and store the frames and joints we need
        frame_joint_map = {}
        for read_line in f:
            #Convert line to list of floats
            processed_rl = [value for value in read_line[:-1].split(' ') if value != '']
            line = [float(i) for i in processed_rl] 

            #If not in map, add it
            if(line[0] not in frame_joint_map): 
                frame_joint_map[line[0]] = []
            
            #Add joints
            frame_joint_map[line[0]].append([line[2],line[3],line[4]])

        #Go through each frame and check if nan values exist; if so, delete
        delete_frames = []
        for frame in sorted(frame_joint_map):
            for j in frame_joint_map[frame]:
                if(True in np.isnan(j)):
                    delete_frames.append(frame)
        delete_frames = set(delete_frames)
        for del_frame in delete_frames:
            del frame_joint_map[del_frame]
        
        #Close file
        f.close()
        
        #For each frame, compute and store distances and angles
        distances = []
        for frame in sorted(frame_joint_map):
            ref_point = frame_joint_map[frame][0]
            frame_distances = []

            #Go through each joint and calculate distance with reference to body center
            for i in range(1,20):
                temp_val = (frame_joint_map[frame][i][0] - ref_point[0])**2
                temp_val2 = (frame_joint_map[frame][i][1] - ref_point[1])**2
                temp_val3 = (frame_joint_map[frame][i][2] - ref_point[2])**2
                d = math.sqrt(temp_val + temp_val2 + temp_val3)
                frame_distances.append(d)
            distances.append(frame_distances)

        #Remove distance outliers
        d1 = remove_outliers(np.array(distances)[:,0])
        d2 = remove_outliers(np.array(distances)[:,1])
        d3 = remove_outliers(np.array(distances)[:,2])
        d4 = remove_outliers(np.array(distances)[:,3])
        d5 = remove_outliers(np.array(distances)[:,4])
        d6 = remove_outliers(np.array(distances)[:,5])
        d7 = remove_outliers(np.array(distances)[:,6])
        d8 = remove_outliers(np.array(distances)[:,7])
        d9 = remove_outliers(np.array(distances)[:,8])
        d10 = remove_outliers(np.array(distances)[:,9])
        d11 = remove_outliers(np.array(distances)[:,10])
        d12 = remove_outliers(np.array(distances)[:,11])
        d13 = remove_outliers(np.array(distances)[:,12])
        d14 = remove_outliers(np.array(distances)[:,13])
        d15 = remove_outliers(np.array(distances)[:,14])
        d16 = remove_outliers(np.array(distances)[:,15])
        d17 = remove_outliers(np.array(distances)[:,16])
        d18 = remove_outliers(np.array(distances)[:,17])
        d19 = remove_outliers(np.array(distances)[:,18])

        #Create distance histograms and normalize
        hist_d1 = np.histogram(d1, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d1))
        hist_d2 = np.histogram(d2, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d2))
        hist_d3 = np.histogram(d3, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d3))
        hist_d4 = np.histogram(d4, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d4))
        hist_d5 = np.histogram(d5, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d5))
        hist_d6 = np.histogram(d6, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d6))
        hist_d7 = np.histogram(d7, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d7))
        hist_d8 = np.histogram(d8, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d8))
        hist_d9 = np.histogram(d9, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d9))
        hist_d10 = np.histogram(d10, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d10))
        hist_d11 = np.histogram(d11, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d11))
        hist_d12 = np.histogram(d12, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d12))
        hist_d13 = np.histogram(d13, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d13))
        hist_d14 = np.histogram(d14, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d14))
        hist_d15 = np.histogram(d15, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d15))
        hist_d16 = np.histogram(d16, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d16))
        hist_d17 = np.histogram(d17, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d17))
        hist_d18 = np.histogram(d18, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d18))
        hist_d19 = np.histogram(d19, dis_bins, range=(lower_bound_dis,upper_bound_dis))[0] / float(len(d19))

        #Concatenate histograms
        output = np.concatenate((hist_d1,hist_d2,hist_d3,hist_d4,hist_d5,
                                    hist_d6,hist_d7,hist_d8,hist_d9,hist_d10,
                                    hist_d11,hist_d12,hist_d13,hist_d14,hist_d15,
                                    hist_d16,hist_d17,hist_d18,hist_d19))
        
        #Print to file
        f = open(output_file,'a')

        if(files[file_index][0:3] == 'a08'):
            f.write('8 ')
        elif(files[file_index][0:3] == 'a10'):
            f.write('10 ')
        elif(files[file_index][0:3] == 'a12'):
            f.write('12 ')
        elif(files[file_index][0:3] == 'a13'):
            f.write('13 ')
        elif(files[file_index][0:3] == 'a15'):
            f.write('15 ')
        elif(files[file_index][0:3] == 'a16'):
            f.write('16 ')

        for i in range(len(output)):
            if(i == len(output)-1):
                f.write(str(i+1) + ':' + str(output[i]) + '\n')
            else: 
                f.write(str(i+1) + ':' + str(output[i]) + ' ')
        f.close()

def process_file_hod(directory, files, output_file):
    #Histogram hyperparameters
    bins = 12

    #For each instance in train or test
    for file_index in range(len(files)):

        #Open file
        file_path = directory + files[file_index]
        f = open(file_path, 'r')
        
        #Read each line in file and store the frames and joints we need
        frame_joint_map = {}
        for read_line in f:
            #Convert line to list of floats
            processed_rl = [value for value in read_line[:-1].split(' ') if value != '']
            line = [float(i) for i in processed_rl] 

            #If not in map, add it
            if(line[0] not in frame_joint_map): 
                frame_joint_map[line[0]] = []
            
            frame_joint_map[line[0]].append([line[2],line[3],line[4]])

        #Go through each frame and check if nan values exist; if so, delete
        delete_frames = []
        for frame in sorted(frame_joint_map):
            for j in frame_joint_map[frame]:
                if(True in np.isnan(j)):
                    delete_frames.append(frame)
        delete_frames = set(delete_frames)
        for del_frame in delete_frames:
            del frame_joint_map[del_frame]
        
        #Close file
        f.close()
        
        output = []

        #Go through all joints
        for joint_num in range(20):
            
            #Create projections 
            XY_projections = []
            XZ_projections = []
            YZ_projections = []
            for frame in sorted(frame_joint_map):
                if(frame == 42):
                    break
                XY_projections.append([frame_joint_map[frame][joint_num][0],frame_joint_map[frame][joint_num][1]])
                XZ_projections.append([frame_joint_map[frame][joint_num][0],frame_joint_map[frame][joint_num][2]])
                YZ_projections.append([frame_joint_map[frame][joint_num][1],frame_joint_map[frame][joint_num][2]])
            
            xy_hist = np.zeros(bins)
            xz_hist = np.zeros(bins)
            yz_hist = np.zeros(bins)

            xy_count, xz_count, yz_count = 0, 0, 0

            #Calculate histograms
            for index in range(len(XY_projections)-1):
                #xy histogram
                delta_rise = (XY_projections[index+1][1]-XY_projections[index][1])
                delta_run = (XY_projections[index+1][0]-XY_projections[index][0])
                if(delta_run != 0):
                    slope = delta_rise / delta_run
                    angle = np.degrees(np.arctan(slope))
                    bin_num = int(math.floor(angle * bins / 360))
                    xy_hist[bin_num] += math.sqrt(delta_rise**2 + delta_run**2)
                    xy_count += 1

                #xz histogram
                delta_rise = (XZ_projections[index+1][1]-XZ_projections[index][1])
                delta_run = (XZ_projections[index+1][0]-XZ_projections[index][0])
                if(delta_run != 0):
                    slope = delta_rise / delta_run
                    angle = np.degrees(np.arctan(slope))
                    bin_num = int(math.floor(angle * bins / 360))
                    xz_hist[bin_num] += math.sqrt(delta_rise**2 + delta_run**2)
                    xz_count += 1

                #yz histogram
                delta_rise = (YZ_projections[index+1][1]-YZ_projections[index][1])
                delta_run = (YZ_projections[index+1][0]-YZ_projections[index][0])
                if(delta_run != 0):
                    slope = delta_rise / delta_run
                    angle = np.degrees(np.arctan(slope))
                    bin_num = int(math.floor(angle * bins / 360))
                    yz_hist[bin_num] += math.sqrt(delta_rise**2 + delta_run**2)
                    yz_count += 1

                if(xy_count != 0):
                    xy_hist = xy_hist / float(xy_count)

                if(xz_count != 0):
                    xz_hist = xz_hist / float(xz_count)

                if(yz_count != 0):
                    yz_hist = yz_hist / float(yz_count)

                output.append(list(xy_hist))
                output.append(list(xz_hist))
                output.append(list(yz_hist))
            
            
        output = np.array(output).flatten()

        #Print to file
        f = open(output_file,'a')

        if(files[file_index][0:3] == 'a08'):
            f.write('8 ')
        elif(files[file_index][0:3] == 'a10'):
            f.write('10 ')
        elif(files[file_index][0:3] == 'a12'):
            f.write('12 ')
        elif(files[file_index][0:3] == 'a13'):
            f.write('13 ')
        elif(files[file_index][0:3] == 'a15'):
            f.write('15 ')
        elif(files[file_index][0:3] == 'a16'):
            f.write('16 ')

        for i in range(len(output)):
            if(i == len(output)-1):
                f.write(str(i+1) + ':' + str(output[i]) + '\n')
            else: 
                f.write(str(i+1) + ':' + str(output[i]) + ' ')
        f.close()

def compute_svm(train_file, test_file, output_file, rep):
    #Change parameters based on representation
    svm_gamma, svm_C = None, None
    if(rep == 'RAD'):
        svm_gamma = 0.5
        svm_C = 0.03125
    elif(rep == 'HJPD'):
        svm_gamma = 0.5
        svm_C = 2.0
    elif(rep == 'HOD'):
        svm_gamma = 32.0
        svm_C = 128
        

    #Classify using svm
    y, x = svm_read_problem('./' + train_file)
    m = svm_train(y,x,'-c ' + str(svm_C) + ' -g ' + str(svm_gamma))
    y, x = svm_read_problem('./' + test_file)
    p_label, p_acc, p_val = svm_predict(y, x, m)
    ACC, MSE, SCC = evaluations(y, p_label)

    #Write predictions to file
    f = open(output_file,'w')
    f.close()
    f = open(output_file, 'a')
    for i in range(len(p_label)):
        f.write(str(int(p_label[i])) + '\n')

    #Compute confusion matrix
    pos_labels = [8.0,10.0,12.0,13.0,15.0,16.0]
    conf_matrix = np.zeros((len(pos_labels),len(pos_labels)))
    for i in range(len(p_label)):
        conf_matrix[pos_labels.index(p_label[i])][pos_labels.index(y[i])] += 1
    
    print("\nConfusion Matrix: ")
    print(conf_matrix)
    print("\nAccuracy: " + str(ACC) + "%")


def main():

    argu = sys.argv[1]

    #Set files 
    if(argu == 'RAD'):
        train_file = 'rad_d2'
        test_file = 'rad_d2.t'
        output_file = 'rad_d2.t.predict'
    elif(argu == 'HJPD'):
        train_file = 'hjpd_d2'
        test_file = 'hjpd_d2.t'
        output_file = 'hjpd_d2.t.predict'
    elif(argu == 'HOD'):
        train_file = 'hod_d2'
        test_file = 'hod_d2.t'
        output_file = 'hod_d2.t.predict'

    #Make (training) file
    directory = './dataset/train/'
    files = sorted(os.listdir(directory))
    f = open(train_file,'w')
    f.close()
    if(argu == 'RAD'):
        process_file_rad(directory, files, train_file)
    elif(argu == 'HJPD'):
        process_file_hjpd(directory, files, train_file)
    elif(argu == 'HOD'):
        process_file_hod(directory, files, train_file)
    

    #Make (testing) file
    directory = './dataset/test/'
    files = sorted(os.listdir(directory))
    f = open(test_file,'w')
    f.close()
    if(argu == 'RAD'):
        process_file_rad(directory, files, test_file)
    elif(argu == 'HJPD'):
        process_file_hjpd(directory, files, test_file)
    elif(argu == 'HOD'):
        process_file_hod(directory, files, test_file)

    #Classify using SVM using RAD representation
    compute_svm(train_file,test_file, output_file, argu)
    

if __name__ == '__main__':
    main()
