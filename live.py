import numpy as np
from scipy import interpolate,signal
import matplotlib.pyplot as plt
import math
from scipy.io import wavfile
import timeit



class FDSAF:
    def __init__( self,filterlen):
        self.M = filterlen
        #self.w_f = np.fft.fft(np.concatenate((np.ones(1),np.zeros(2*self.M-1))))
        self.w_f = np.fft.fft(np.zeros(2*self.M))
        #self.w_f = np.fft.fft(np.concatenate((np.ones(self.M)/self.M,np.zeros(self.M))))
        self.last_buffer = np.zeros(self.M, dtype='float')
        self.Cm = np.matrix(0.5 * np.array([[-1,3,-3,1], # Row major
                                 [2,-5,4,-1],
                                 [-1,0,1,0],
                                 [0,2,0,0]]))
        
        # Based on paper example 1                       
        self.mu_w = 0.001
        self.mu_q = 0.001
        self.delta_x = 0.2
        self.ordinates = np.arange(-2.2,2.3,self.delta_x)
        self.abscissas = np.arange(-2.2,2.3,self.delta_x) # Only for graphing

        self.N = len(self.ordinates)-1
        self.e = 0

        # For sample-wise filtering
        #self.w = np.concatenate((np.ones(1),np.zeros(self.M-1)))
        self.w = np.zeros(self.M)
        self.single_buffer = np.zeros(self.M)


    def est(self, x, d):
        if len(x) != self.M or len(d) != self.M:
            print("Wrong input length")
            exit()

        full_buffer = np.concatenate((self.last_buffer,x))

        x_f = np.fft.fft(full_buffer)

        s = np.fft.ifft(x_f*self.w_f)[-self.M:]

        UT = []
        UdotT = []
        QT = []
        IT = []

        for j in range(self.M):
            i_j = int(np.floor(s[j].real/self.delta_x) + (self.N/2))
            u_j = s[j].real/self.delta_x - np.floor(s[j].real/self.delta_x)
            u = [math.pow(u_j,3),math.pow(u_j,2),u_j,1]
            udot = [3*math.pow(u_j,2),2*u_j,1,0]

            if (abs(s[j])>2.0):
                q = np.asarray([(y-11)*self.delta_x for y in range(i_j-1,i_j+3)])
                IT.append([-1,-1,-1,-1])
            else:
                q = np.asarray(list(self.ordinates[i_j-1:i_j+3]))
                IT.append([i_j-1,i_j,i_j+1,i_j+2])

            UT.append(u)
            UdotT.append(udot)
            QT.append(q)

        Um = np.matrix(UT).T
        Udotm = np.matrix(UdotT).T
        Qm = np.matrix(QT).T
        Im = np.matrix(IT).T

        y = np.matmul(self.Cm,Qm)
        y = np.multiply(y, Um)
        y = np.asarray(y)
        y = np.sum(y, axis=0)

        self.e = d - y

        deltadot = np.matmul(self.Cm,Qm)
        deltadot = np.multiply(deltadot,Udotm)
        deltadot = np.asarray(deltadot)
        deltadot = np.sum(deltadot,axis=0)

        e_s = np.multiply(deltadot/self.delta_x,self.e)
        e_f = np.fft.fft(np.concatenate((np.zeros(self.M),e_s)))
        deltaW = np.fft.ifft(np.multiply(e_f,np.conjugate(x_f)))[:self.M]

        self.w_f = self.w_f + self.mu_w * np.fft.fft(np.concatenate((deltaW,np.zeros(self.M))))

        temp = np.asarray(np.matmul(self.Cm.T,Um))
        deltaq = self.mu_q * self.e * temp
        Qm = Qm + deltaq
        Qm = np.reshape(Qm,-1)
        Im = np.reshape(Im,-1)
        deltaq = np.reshape(deltaq,-1)

        for i in range(np.shape(Im)[1]):
            if Im[0,i] != -1:
                self.ordinates[Im[0,i]] += deltaq[i]

        self.last_buffer = x
        return y

    def estsingle(self, x, d):

        # Filter
        self.single_buffer[1:] = self.single_buffer[:-1]
        self.single_buffer[0] = x
        s = np.dot(self.single_buffer,self.w)

        i_j = int(np.floor(s/self.delta_x) + (self.N/2))

        u_j = s/self.delta_x - np.floor(s/self.delta_x)

        u = np.asarray([math.pow(u_j,3),math.pow(u_j,2),u_j,1])
        udot = np.asarray([3*math.pow(u_j,2),2*u_j,1,0])

        if (abs(s)>2.0):
            q = np.asarray([(y-11)*self.delta_x for y in range(i_j-1,i_j+3)])
        else:
            q = np.asarray(list(self.ordinates[i_j-1:i_j+3]))

        y = np.matmul(u,self.Cm)
        y = np.matmul(y, q)
        y = float(y)

        self.e = d - y

        # Train filter
        deltaw = np.matmul(self.mu_w*self.e*udot,self.Cm)
        deltaw = float(np.matmul(deltaw, q))
        self.w = self.w + deltaw*self.single_buffer

        if abs(s) <= 2.0:
            deltaq = np.matmul(self.mu_q*self.e*self.Cm.T,u)
            q = q + deltaq
            for i in range(np.shape(q)[1]):
                self.ordinates[i_j+i-1] = float(q[0,i])

        return y

    def estsinglefilter(self, x, d):

        # Filter
        self.single_buffer[1:] = self.single_buffer[:-1]
        self.single_buffer[0] = x
        # print(self.w)
        s = np.dot(self.single_buffer,self.w)


        i_j = int(np.floor(s/self.delta_x) + (self.N/2))
        q = np.asarray([(y-11)*self.delta_x for y in range(i_j-1,i_j+3)])

        u_j = s/self.delta_x - np.floor(s/self.delta_x)
        u = np.asarray([math.pow(u_j,3),math.pow(u_j,2),u_j,1])
        udot = np.asarray([3*math.pow(u_j,2),2*u_j,1,0])

        y = np.matmul(u,self.Cm)
        y = np.matmul(y, q)
        
        y = float(y)

        self.e = d - y

        # Train filter
        deltaw = np.matmul(self.mu_w*self.e*udot,self.Cm)
        deltaw = float(np.matmul(deltaw, q))
        self.w = self.w + deltaw*self.single_buffer

        # print(self.w[0])
        return y

    def estsinglespline(self, x, d):
        i_j = int(np.floor(x/self.delta_x) + (self.N/2))

        u_j = x/self.delta_x - np.floor(x/self.delta_x)

        u = np.asarray([math.pow(u_j,3),math.pow(u_j,2),u_j,1])

        q = np.asarray(list(self.ordinates[i_j-1:i_j+3]))

        y = np.matmul(u,self.Cm)
        y = np.matmul(y, q)
        y = float(y)

        self.e = d - y

        deltaq = np.matmul(self.mu_q*self.e*self.Cm.T,u)
        q = q + deltaq

        for i in range(np.shape(q)[1]):
            self.ordinates[i_j+i-1] = float(q[0,i])

        return y

if __name__ == "__main__":

    sr, data = wavfile.read("4065long.wav")

    speaker1 = data[:,0]
    speaker2 = data[:,1]

    echo = np.copy(speaker1)

    echo_amp = 0.5 # ratio
    echo_delay = 0.1 #seconds

    roll_amt = int(echo_delay * sr)

    echo = signal.lfilter(signal.firwin(50,0.3,pass_zero='lowpass'),1,echo)

    echo = echo * echo_amp
    echo = np.roll(echo,roll_amt)
    echo[:roll_amt] = np.zeros(roll_amt)

    speaker2wecho = speaker2+echo

    output = np.asarray([[speaker1[i],speaker2wecho[i]] for i in range(len(speaker1)) ])
    wavfile.write("bad.wav",sr,output)

    x = speaker1
    d = speaker2wecho

    filterlen = int(roll_amt*1.5)

    FD = FDSAF(filterlen)

    y = []
    e = []


    do_fdsaf = True
    if do_fdsaf:
        # Perform filtering with FDSAF
        zext_len = filterlen - (len(x)%filterlen) if len(x)%filterlen !=0 else 0
        x = np.pad(x,(0,zext_len),'constant')
        x = np.reshape(x,(-1,filterlen))
        d = np.pad(d,(0,zext_len),'constant')
        d = np.reshape(d,(-1,filterlen))
        for k in range(len(x)):
            y.append(FD.est(x[k], d[k]))
            e.append(FD.e)
        y = np.asarray(y).flatten()
        e = np.asarray(e).flatten()
        e = e[:-zext_len]
    else:
        # Perform normal SAF
        for k in range(len(x)):
            y.append(FD.estsingle(x[k], d[k]))
            e.append(FD.e)


    ERLE = signal.lfilter(np.ones(10000)/10000,1,20*np.log(abs(echo/(e-speaker2))))
    plt.title("Echo vs ERLE")
    ax = plt.subplot(111)
    ax.plot(echo,label="Echo")
    ax2 = plt.twinx(ax)
    ax2.plot(ERLE,'r',label="ERLE")
    plt.legend()
    ax.legend(loc="lower left")
    plt.show()
    



    plt.show()
    plt.title("FIR Filter Weight")
    plt.ylabel("Filter Weight")
    plt.xlabel("Tap")
    plt.plot(np.fft.fft(FD.w_f).real[:int(len(FD.w_f)/2):-1]/240)
    plt.show()

    plotx = np.linspace(-2.2,2.2,100000)
    resultspline = interpolate.splrep(FD.abscissas, FD.ordinates, s=0)
    resulty = interpolate.splev(plotx, resultspline, der=0)

    plt.plot(plotx,resulty)

    plt.title("Result Spline")

    plt.show()

    output = np.asarray([[speaker1[i],e[i]] for i in range(len(speaker1)) ])
    wavfile.write("good.wav",sr,output)

    residual = e - speaker2
    wavfile.write("residualecho.wav",sr,residual)

    comparison = np.asarray([[echo[i], residual[i]] for i in range(len(e))])
    wavfile.write("compare.wav",sr,comparison)



