import numpy as np
from scipy import interpolate,signal
import matplotlib.pyplot as plt
import math


class FDSAF:
    def __init__( self,filterlen ):
        self.M = filterlen
        self.w_f = np.fft.fft(np.concatenate((np.ones(1),np.zeros(2*self.M-1))))
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
        self.w = np.concatenate((np.ones(1),np.zeros(self.M-1)))
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


        # y = np.matmul(Um.T,self.Cm)
        # y = np.multiply(y,Qm.T)
        # y = np.asarray(y)
        # y = np.sum(y,axis=1)

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
        Qm = Qm + deltaq # Now need to update these in the abscissa array
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
        # print(self.w)
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
    np.random.seed(0)
    filterlen = 120
    FD = FDSAF(filterlen) # initialize our FDSAF

    y = []
    e = []

    # FILTER COEFFICIENTS
    a = [1,-2.628,2.3,-0.6703]
    b = [0,0.1032,-0.0197,-0.0934]

    # SPLINE KNOTS
    spline_y = [-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,
                -0.91,-0.42,-0.01,-0.1,0.1,-0.15,0.58,1.2,
                1.0,1.2,1.4,1.6,1.8,2.0,2.2]
    spline_x = np.linspace(-2.2,2.2,len(spline_y))
    targetspline = interpolate.splrep(spline_x, spline_y, s=0)

    ### TEST FOR EX 1
    x = np.random.uniform(-1,1,100000)
    d = signal.lfilter(b,a,x)
    d = interpolate.splev(d, targetspline, der=0)

    signal_power = np.mean(d**2)
    d = d + np.random.normal(loc=0.0,scale= signal_power/1000,size=len(d))

     #FOR BLOCKWISE TESTING
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

    plotx = np.linspace(-2.2,2.2,100000)
    targety = interpolate.splev(plotx, targetspline, der=0)
    resultspline = interpolate.splrep(FD.abscissas, FD.ordinates, s=0)
    resulty = interpolate.splev(plotx, resultspline, der=0)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax2 = plt.twiny(ax)
    ax2.plot(targety,label="Target Spline")
    ax2.get_xaxis().set_visible(False)
    ax.plot(spline_x,spline_y,'or',label="Spline Knots")
    ax2.plot(resulty,label="Result Spline")
    ax.legend()
    ax2.legend(loc="lower right")
    plt.title("Spline")
    plt.show()

    plt.title("Mean Squared Error")
    plt.xlabel("Sample")
    plt.ylabel("Mean Squared Error")

    plt.plot(signal.lfilter(np.ones(100)/100,1,np.asarray(e)**2))
    plt.show()

    neww = np.fft.fft(FD.w_f).real[:int(len(FD.w_f)/2):-1]
    neww = neww/240

    # plt.plot(neww)
    # plt.show()
    # exit()

    wprime,hprime = signal.freqz(b,a)
    w, h = signal.freqz(neww)
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)
   
    ax1.plot(w, 20 * np.log10(abs(h)), 'b',label="Result Amplitude")
    ax1.plot(wprime, 20 * np.log10(abs(hprime)), 'r',label="Target Amplitude")
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
   
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    anglesprime = np.unwrap(np.angle(hprime))
    ax2.plot(w, angles, 'g',label="Result Phase")
    ax2.plot(wprime,anglesprime, 'm',label="Target Phase")
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    ax1.legend()
    ax2.legend(loc="lower left")
    plt.show()

    # ### WORKING TEST TO SHOW SAMPLE LEVEL FILTER AND SPLINE WITH RANDOM INPUT (WITH NOISE)

    # x = np.random.uniform(-1,1,100000)
    # d = signal.lfilter(b,a,x)
    # d = interpolate.splev(d, targetspline, der=0)

    # signal_power = np.mean(d**2)
    # d = d + np.random.normal(loc=0.0,scale= signal_power/1000,size=len(d))

    # for k in range(np.shape(x)[0]):
    #     y.append(FD.estsingle(x[k], d[k]))
    #     e.append(FD.e)

    # plotx = np.linspace(-2.2,2.2,100000)
    # targety = interpolate.splev(plotx, targetspline, der=0)
    # resultspline = interpolate.splrep(FD.abscissas, FD.ordinates, s=0)
    # resulty = interpolate.splev(plotx, resultspline, der=0)

    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax2 = plt.twiny(ax)
    # ax2.plot(targety,label="Target Spline")
    # ax2.get_xaxis().set_visible(False)
    # ax.plot(spline_x,spline_y,'or',label="Spline Knots")
    # ax2.plot(resulty,label="Result Spline")
    # ax.legend()
    # ax2.legend(loc="lower right")
    # plt.title("Spline")
    # plt.show()

    # plt.title("Mean Squared Error")
    # plt.xlabel("Sample")
    # plt.ylabel("Mean Squared Error")

    # plt.plot(signal.lfilter(np.ones(100)/100,1,np.asarray(e)**2))
    # plt.show()


    # wprime,hprime = signal.freqz(b,a)
    # w, h = signal.freqz(FD.w)
    # fig = plt.figure()
    # plt.title('Digital filter frequency response')
    # ax1 = fig.add_subplot(111)
   
    # ax1.plot(w, 20 * np.log10(abs(h)), 'b',label="Result Amplitude")
    # ax1.plot(wprime, 20 * np.log10(abs(hprime)), 'r',label="Target Amplitude")
    # plt.ylabel('Amplitude [dB]', color='b')
    # plt.xlabel('Frequency [rad/sample]')
   
    # ax2 = ax1.twinx()
    # angles = np.unwrap(np.angle(h))
    # anglesprime = np.unwrap(np.angle(hprime))
    # ax2.plot(w, angles, 'g',label="Result Phase")
    # ax2.plot(wprime,anglesprime, 'm',label="Target Phase")
    # plt.ylabel('Angle (radians)', color='g')
    # plt.grid()
    # plt.axis('tight')
    # ax1.legend()
    # ax2.legend(loc="lower left")
    # plt.show()


    # ### WORKING TEST TO SHOW SAMPLE LEVEL FILTER WITH RANDOM INPUT

    # x = np.random.uniform(-1,1,100000)
    # d = signal.lfilter(b,a,x)

    # for k in range(np.shape(x)[0]):
    #     y.append(FD.estsinglefilter(x[k], d[k]))
    #     e.append(FD.e)

    # e = np.asarray(e)**2
    # e = signal.lfilter(np.ones(1000)/1000,[1],e)
    # plt.plot(e)
    # plt.show()
    # plt.plot(FD.w)
    # plt.show()
    # print(FD.w)

    # wprime,hprime = signal.freqz(b,a)
    # w, h = signal.freqz(FD.w)
    # fig = plt.figure()
    # plt.title('Digital filter frequency response')
    # ax1 = fig.add_subplot(111)
   
    # plt.plot(w, 20 * np.log10(abs(h)), 'b')
    # plt.plot(wprime, 20 * np.log10(abs(hprime)), 'r')
    # plt.ylabel('Amplitude [dB]', color='b')
    # plt.xlabel('Frequency [rad/sample]')
   
    # ax2 = ax1.twinx()
    # angles = np.unwrap(np.angle(h))
    # anglesprime = np.unwrap(np.angle(hprime))
    # plt.plot(w, angles, 'g')
    # plt.plot(wprime,anglesprime, 'm')
    # plt.ylabel('Angle (radians)', color='g')
    # plt.grid()
    # plt.axis('tight')
    # plt.show()

    # ### WORKING TEST TO SHOW SAMPLE LEVEL SPLINE ADAPTION WITH RANDOM INPUT
    # x = np.random.uniform(-1,1,100000)
    # d = interpolate.splev(x, targetspline, der=0)
    # for k in range(np.shape(x)[0]):
    #     y.append(FD.estsinglespline(x[k], d[k]))
    #     e.append(FD.e)

    # plotx = np.linspace(-2.2,2.2,100000)
    # targety = interpolate.splev(plotx, targetspline, der=0)
    # resultspline = interpolate.splrep(FD.abscissas, FD.ordinates, s=0)
    # resulty = interpolate.splev(plotx, resultspline, der=0)
    # plt.plot(targety)
    # plt.plot(resulty)
    # plt.show()
    # plt.plot(np.asarray(e)**2)
    # plt.show()

    # ### WORKING TEST TO SHOW SAMPLE LEVEL SPLINE ADAPTION WITH LINSPACE
    # x = np.linspace(-1.8,1.8,100000)
    # d = interpolate.splev(x, targetspline, der=0)
    # for k in range(np.shape(x)[0]):
    #     y.append(FD.estsingle(x[k], d[k]))
    #     e.append(FD.e)
    # targetynew = interpolate.splev(x, targetspline, der=0)
    # plt.plot(targetynew)
    # plt.plot(y)
    # plt.show()