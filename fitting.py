import numpy
#import matplotlib.pyplot as plt
import glob
Win = 21
degree = 5
list = []
for i in range(3):
    for j in range(5):
        list.append(1 + j + i * 24)        

SV = [0, 0, 0]
SVS = [0, 0, 0]
ML = [0, 0, 0]
SDL = [0, 0, 0]
SA = [0, 0, 0]
SP = [0, 0, 0]
MX = [0, 0, 0]
SDX = [0, 0, 0]
RanX = [0, 0, 0]
MY = [0, 0, 0]
SDY = [0, 0, 0]
RanY = [0, 0, 0]
MS = [0, 0, 0]
MK = [0, 0, 0]
SSD = [0, 0, 0]
MVX = [0, 0, 0]
MVY = [0, 0, 0]
SVX = [0, 0, 0]
SVY = [0, 0, 0]
MI = [0, 0, 0]
CVR = [0, 0, 0]
CVV = [0, 0, 0]
CC = [0, 0, 0]
CCV = [0, 0, 0]
MPX = [0, 0, 0]
MPY = [0, 0, 0]

files = glob.glob("Sujeto*.csv")
print(files)

for name in files:
    dat = name[:-4] + ".dat"
    print(dat)
    fp = open(name,"r")
    string = [ line.split(";") for line in fp.readlines()]
    fp.close()
    string.pop(0)
    string.pop(0)

    flt = [ [ float(string[i][j]) for i in range(len(string)) ] for j in list ]
    data = numpy.array(flt, order = "F")
    W = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    aux = numpy.array([ 0.0 for i in range(len(data[0])) ])
    X = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    Y = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    FFX = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    FFY = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])    
    fitting = numpy.array([ 0.0 for i in range(Win) ])
    V = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    VY = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    VX = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    L = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    A = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])
    I = numpy.array([ [ 0.0 for i in range(len(data[0])) ] for j in range(3) ])

    fp = open(dat,"w")

    for j in range(3):
        W[j] = data[1 + 5 * j] + data[2 + 5 * j] + data[3 + 5 * j] + data[4 + 5 * j]
        
        Y[j] = data[1 + 5 * j] + data[2 + 5 * j] - data[3 + 5 * j] - data[4 + 5 * j]
        Y[j] /= W[j]
        Y[j] *= 17.5
        MY[j] = numpy.average(Y[j])
        SDY[j] = numpy.std(Y[j])
        RanY[j] = numpy.amax(Y[j]) - numpy.amin(Y[j])
        FFY[j] = numpy.square(numpy.abs(numpy.fft.fft(Y[j])))
        MPY[j] = numpy.median(FFY[j])
        
        X[j] = - data[1 + 5 * j] + data[2 + 5 * j] + data[3 + 5 * j] - data[4 + 5 * j]
        X[j] /= W[j]
        X[j] *= 17.5
        MX[j] = numpy.average(X[j])
        SDX[j] = numpy.std(X[j])
        RanX[j] = numpy.amax(X[j]) - numpy.amin(X[j])
        FFX[j] = numpy.square(numpy.abs(numpy.fft.fft(X[j])))
        MPX[j] = numpy.median(FFX[j])
        
        
        CC[j] = numpy.cov(X[j], Y[j])
        CVR[j] = CC[j][0][1]

        
        M = numpy.vander(data[j * 5][0:Win],degree + 1)
        MT = M.T
        M = MT @ M
        M = numpy.linalg.inv(M)
        M = M @ MT
        
        for i in range(len(data[0]) - Win + 1):
            fitting = X[j][i:i + Win]
            fit = M @ fitting
            dfit = numpy.polyder(fit)
            P = numpy.poly1d(dfit)
            VX[j][10 + i] = P(data[j * 5][10])
            fitting = Y[j][i:i + Win]
            fit = M @ fitting
            dfit = numpy.polyder(fit)
            P = numpy.poly1d(dfit)
            VY[j][10 + i] = P(data[j * 5][10])
            
            
        L[j] = X[j] * VY[j] - Y[j] * VX[j]
        A[j] = numpy.fabs(L[j]) / 2.0
        V[j] = numpy.hypot(VX[j],VY[j])
        
        #fig, axs = plt.subplots()
        #axs.plot(X[j], Y[j])
        #fig.tight_layout()
        #plt.show()
        #fig, axs = plt.subplots()
        #axs.plot(data[5 * j], FFX[j])
        #fig.tight_layout()
        #plt.show()
        #fig, axs = plt.subplots()
        #axs.plot(data[5 * j], FFY[j])
        #fig.tight_layout()
        #plt.show()
            
        SV[j] = numpy.average(A[j])
        SVS[j] = numpy.std(A[j]) 
        SP[j] = (numpy.sum(V[j]) - V[j][10] * 0.5 - V[j][len(data[0])-10] * 0.5) * (data[j * 5][1] - data[j * 5][0])
        SA[j] = (numpy.sum(A[j]) - A[j][10] * 0.5 - A[j][len(data[0])-10] * 0.5) * (data[j * 5][1] - data[j * 5][0])
        ML[j] = numpy.average(L[j])
        SDL[j] = numpy.std(L[j])
        MS[j] = numpy.average(V[j])
        SSD[j] = numpy.std(V[j])
        MVX[j] = numpy.average(VX[j])
        SVX[j] = numpy.std(VX[j])
        MVY[j] = numpy.average(VY[j])
        SVY[j] = numpy.std(VY[j])
        
        CCV[j] = numpy.cov(VX[j], VY[j])
        CVV[j] = CCV[j][0][1]
        MK[j] = (SSD[j] * SSD[j] + MS[j] * MS[j]) / 2
        
        fp.write("# " + "*" * 80 + "\n")
        fp.write("# Filename: %s\n" % name)
        fp.write("# Attempt: %s\n" % (j + 1))
        fp.write("# Savitzky-Golay window: %s\n" % Win)
        fp.write("# Savitzky-Golay degree: %s\n" % degree)
        fp.write("# Sway Area: %s\n" % SA[j])
        fp.write("# Sway Path: %s\n" % SP[j])
        fp.write("# Mean Sway Velocity: %s\n" % SV[j])
        fp.write("# Sway Velocity Std Dev: %s\n" % SVS[j])
        fp.write("# Mean Angular Momentum: %s\n" % ML[j])
        fp.write("# Anglular Momentum Std Dev: %s\n" % SDL[j])
        fp.write("# Mean X: %s\n" % MX[j])
        fp.write("# X Std Dev : %s\n" % SDX[j])
        fp.write("# Ran X: %s\n" % RanX[j])
        fp.write("# Mean Power Frequency X: %s\n" % MPX[j])
        fp.write("# Mean Velocity X: %s\n" % MVX[j])
        fp.write("# VX Std Dev : %s\n" % SVX[j])
        fp.write("# Mean Y: %s\n" % MX[j])
        fp.write("# Y Std Dev : %s\n" % SDY[j])
        fp.write("# Ran Y: %s\n" % RanY[j])
        fp.write("# Mean Power Frequency Y: %s\n" % MPY[j])
        fp.write("# Mean Velocity Y: %s\n" % MVY[j])
        fp.write("# VY Std Dev : %s\n" % SVY[j])
        fp.write("# Mean Speed: %s\n" % MS[j])
        fp.write("# Speed Std Dev: %s\n" % SSD[j])
        fp.write("# Mean Kinetiv Energy: %s\n" % MK[j])
        fp.write("# Position Covariance: %s\n" % CVR[j])
        fp.write("# Velocity Covariance: %s\n" % CVV[j])
        a = "[ " + str(CC[j][0][0]) + " " + str(CC[j][0][1]) + " ]"
        fp.write("# Position Cov Matrix: %s\n" % a)
        a = "[ " + str(CC[j][0][0]) + " " + str(CC[j][0][1]) + " ]"
        fp.write("#                      %s\n" % a)
        a = "[ " + str(CCV[j][0][0]) + " " + str(CCV[j][0][1]) + " ]"
        fp.write("# Velocity Cov Matrix: %s\n" % a)
        a = "[ " + str(CCV[j][1][0]) + " " + str(CCV[j][1][1]) + " ]"
        fp.write("#                      %s\n" % a)
        fp.write("# " + "*" * 80 + "\n")
        fp.write("# Checks must be close to 0\n")
        a = ML[j] * ML[j] + SDL[j] * SDL[j] - 4.0 * (SV[j] * SV[j]) - 4.0 * (SVS[j] * SVS[j])
        fp.write("# Check #1: %s\n" % a)
        a = SSD[j] * SSD[j] + MS[j] * MS[j] - MVX[j] * MVX[j] - SVX[j] * SVX[j] - MVY[j] * MVY[j] - SVY[j] * SVY[j]
        fp.write("# Check #2: %s\n" % a)
        fp.write("# " + "*" * 80 + "\n")
        fp.write("\n")
        
        fp.write("#t;X;Y;VX;VY;L\n")
        for i in range(len(data[0])):
            a = str(data[j * 5][i]) + ";" + str(X[j][i]) + ";" + str(Y[j][i]) + ";" + str(VX[j][i]) + ";" + str(VY[j][i]) + ";" + str(L[j][i])
            fp.write("%s\n" % a)
        fp.write("\n")
        fp.write("\n")

    
    fp.close()

    print("SA: ", SA)
    print("SP: ", SP)
    print("SV: ", SV)
    print("SVS: ", SVS)
    print("ML: ", ML)
    print("SDL: ", SDL)
    print("MX:", MX)
    print("SDX: ", SDX)
    print("RanX: ", RanX)
    print("MPX: ", MPX)
    print("MVX: ", MVX)
    print("SVX: ", SVX)
    print("MY: ", MY)
    print("SDY: ", SDY)
    print("RanY: ", RanY)
    print("MPY: ", MPY)
    print("MVY: ", MVY)
    print("SVY: ", SVY)
    print("MS: ", MS)
    print("SSS: ", SSD)
    print("MK: ", MK)
    print("CCR: ", CVR)
    print("CVV: ", CVV)
    print("CC: ", CC)
    print("CCV: ", CCV)
