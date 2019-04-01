
# coding: utf-8

# In[ ]:

import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import odeint
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.integrate as integrate
from matplotlib.animation import FuncAnimation


class ParticleSys:

    #def __init__(self,system_particleP,system_particleW,sys_row,sys_col):
    def __init__(self,sys_row,sys_col,dtime,total_part,nInjPs,sys_len,sys_wid):
        self.Me=9.10938356E-31
        self.En=0.1*1.6E-16
        self.q=1.6e-19  
        self.t = 0.0                 # Start of integration
        self.tStop = 7.545e-5             # End of integration    
        #dtime = 3.97105263e-06 
        self.dtime = dtime 
        self.system_length = sys_len
        self.system_height = sys_wid
        self.E_field=np.array([0,1.98E-3,0])    #Electric field in the domain in the ydirection
        self.x0=0
        self.y0=0
        self.R=np.array([1,1])
        self.total_particle = total_part
        self.N_inj_perStep = nInjPs
        self.sys_row = sys_row
        self.sys_col = sys_col
        self.low_row  = 0
        self.high_row = self.N_inj_perStep
        self.current_inj = 0
        self.total_inj = 0
        self.np_out = 0
        #create the particle system
        self.system_particleP = np.zeros((sys_row,sys_col))
        self.system_particleW = np.zeros((sys_row,1),dtype=np.int)

        self.initial = True
        self.counter = 1
        self.steadycount = 0
        ##self.state = pd.DataFrame()
      




    def BField(self,x,y):
        B_1=np.array([0,0,4.01E-9])
        B_2=np.array([0,0,10.82E-7])
        r0=np.array([self.x0,self.y0])
        r=np.array([x,y])
        if np.dot((r-r0),(r-r0))<np.dot((self.R-r0),(self.R-r0)):
            B=B_2
        else:
             B=B_1
        return B

    def ele_traj(self,w,t,q,Me):
        xe_comp,ye_comp,ze_comp,vex_comp,vey_comp,vez_comp=w
        v_vect=np.array([vex_comp,vey_comp,vez_comp])
        v_cro_B=np.cross(v_vect,self.BField(xe_comp,ye_comp))
        force=(-self.q/self.Me)*(self.E_field+v_cro_B)
        dwdt=[vex_comp,vey_comp,vez_comp,force[0],force[1],force[2]]
        return dwdt

    def find_position(self,high):
        for row in range(0,high):
            vxe0_comp=self.system_particleP[row,3]         # x co-ordinate of the velocicy of the electron
            vye0_comp=self.system_particleP[row,4]        # y co-ordinate of the velocicy of the electron
            vze0_comp=self.system_particleP[row,5]         # z co-ordinate of the velocicy of the electron
            xe0_comp=self.system_particleP[row,0]                          # x co-ordinate of the initial condition 
            ye0_comp=self.system_particleP[row,1]                    # y co-ordinate of the initial condition
            ze0_comp=self.system_particleP[row,2]         # z co-ordinate of the initial condition
            w0=[xe0_comp,ye0_comp,ze0_comp,vxe0_comp,vye0_comp,vze0_comp]
            #print(w0)
            t = np.linspace(0, self.dtime, 100)
            sol_electr=odeint(self.ele_traj,w0,t,args=(self.q,self.Me))
            xnew = sol_electr[-1,0]
            ynew = sol_electr[-1,1]
            znew = sol_electr[-1,2]
            vxnew = sol_electr[-1,3]
            vynew = sol_electr[-1,4]
            vznew = sol_electr[-1,5]

          
        
            if xnew < -(self.system_length/2):
                vxnew=vxnew*-1
                xnew=-xnew+2*(-self.system_length/2)
            if ynew>(self.system_height/2):
                ynew=2*(self.system_height/2)-ynew
                vynew*=-1
            if ynew<-(self.system_height/2):
                ynew=-ynew+2*(-self.system_height/2)
                vynew*=-1
            if ynew>(self.system_height/2) and xnew<(-self.system_length/2):
                vynew*=-1
                xnew=-xnew+2*(-self.system_length/2)
                ynew=2*(self.system_height/2)-ynew
                vxnew=vxnew*-1
            if ynew<(-self.system_height/2) and xnew<(-self.system_length/2):
                vynew*=-1
                xnew=-xnew+2*(-self.system_length/2)
                ynew=-ynew+2*(-self.system_height/2)
                vxnew=vxnew*-1
            if xnew > (self.system_length/2):
                self.system_particleW[row] = 0
            else:
                self.system_particleW[row] = 1
            #update system particle

            self.system_particleP[row,0] = xnew
            self.system_particleP[row,1] = ynew
            self.system_particleP[row,2] = znew
            self.system_particleP[row,3] = vxnew
            self.system_particleP[row,4] = vynew
            self.system_particleP[row,5] = vznew


    def ReplaceOut(self):
        #change this to loop over available particles 
        for ind in range(self.total_inj):
            if self.system_particleW[ind] == 0:
                print("<ReplaceOut> replacing: index",ind)
                print(ind,self.system_particleW[ind])
                self.system_particleP[ind,0]=np.random.uniform((-self.system_length/2),(-self.system_length/5)) 
                self.system_particleP[ind,1]=np.random.uniform((-self.system_height/2),(self.system_height/2)) 
                self.system_particleP[ind,2]=0
                #velecity Vx,Vy,Vz index(3,4,5)
                self.system_particleP[ind,3]=np.sqrt(self.En/self.Me)*sum(np.random.rand(12)-0.5)
                self.system_particleP[ind,4]=np.sqrt(self.En/self.Me)*sum(np.random.rand(12)-0.5)
                self.system_particleP[ind,5]=0
                #set weight to 1
                self.system_particleW[ind]=1

    def rangeParticleInj(self):
        print("<rangeParticleInj> Injecting %d particles" % (self.current_inj))
        for col in range(self.sys_col):
            for row in range(self.low_row,self.high_row):
                ##print("index",row,col)
                #position X,Y,Z (index 0,1,2)
                if col == 0:
                    self.system_particleP[row,col]=np.random.uniform((-self.system_length/2),(-self.system_length/5)) 
                if col == 1:
                    self.system_particleP[row,col]=np.random.uniform((-self.system_height/2),(self.system_height/2)) 
                if col == 2:
                    self.system_particleP[row,col]=0
                #velecity Vx,Vy,Vz index(3,4,5)
                if col == 3:
                    self.system_particleP[row,col]=np.sqrt(self.En/self.Me)*sum(np.random.rand(12)-0.5)
                if col == 4:
                    self.system_particleP[row,col]=np.sqrt(self.En/self.Me)*sum(np.random.rand(12)-0.5)
                if col == 5:
                    self.system_particleP[row,col]=0
                self.system_particleW[row]=1

    def getNParticleOutside(self):
        ##print("<getNParticleOutside> total num Par in Sys ",self.total_inj)
        sumW = np.sum(self.system_particleW,dtype=int)
        ##print("<getNParticleOutside> num par inside sys: ",sumW)
        out = self.total_inj - sumW 
        return out
    def printSys_dataframe(self):
        col_lab1=['X','Y','Z','Vx','Vy','Vz']
        col_lab2=['W']
        #dataframe of position, weight and merge all 
        df_pos = pd.DataFrame(self.system_particleP,columns=col_lab1)
        df_weight = pd.DataFrame(self.system_particleW,columns=col_lab2)
        df_particle = pd.concat([df_pos,df_weight],axis=1)
        print(df_particle)

    def drawposition(self):
        ##print("<drawposition>: ",self.total_inj)
        col_lab1=['X','Y','Z','Vx','Vy','Vz']
        col_lab2=['W']
        df_pos = pd.DataFrame(self.system_particleP,columns=col_lab1)
        df_weight = pd.DataFrame(self.system_particleW,columns=col_lab2)
        df_particle = pd.concat([df_pos,df_weight],axis=1)
 
        plt.clf()
        sns.scatterplot(x="X", y="Y",hue="W", data=df_particle[:self.total_inj])
        #plt.savefig('plots/plots_'+str(counter) +'.pdf')
        plt.savefig('plots/plots_gg.pdf')

    def step(self):
        fileout = open("out_class_file.txt","a")
        print("init pos",self.system_particleP)
        print("int weight",self.system_particleW)

        ###while True:
        print("row_low",self.low_row)
        print("row_high",self.high_row)
        self.current_inj = 0
        self.np_out  = 0
        if self.initial == True:
            self.total_inj+=self.N_inj_perStep
            self.current_inj+=self.N_inj_perStep
            self.rangeParticleInj()
            ##self.printSys_dataframe()
            self.find_position(self.total_inj)
            self.printSys_dataframe()

            print("step: %d curent inj: %s n out: %s, total injected %s "%(self.counter,self.current_inj,0,self.total_inj))
            fileout.write("%d,%2.6f,%d\n" % (self.counter,self.dtime*self.counter,self.total_inj))
            fileout.flush()
            self.initial=False
            self.counter+=1
            #continue
        self.np_out = self.getNParticleOutside()
        print("np out: ",self.np_out)
        if self.np_out < 1:
            self.current_inj = self.N_inj_perStep
            self.total_inj+=self.current_inj
            self.low_row  = self.high_row
            self.high_row = self.high_row+self.current_inj
            print("cur inj: %d, low row %d, high row %d"%(self.current_inj,self.low_row,self.high_row))
            self.rangeParticleInj()
            ##self.printSys_dataframe()
            self.find_position(self.total_inj)
                
            self.printSys_dataframe()
        #else if particle are out bound 
        # drop and initialize the nout particles outbound 
        # add another     n_inj_perStep - nout (it is posible to have -ve)
        else:
            #Note: this replace np_out (zero weight)
            self.ReplaceOut() 
            ##self.printSys_dataframe()
            
            #add the rest of the particle
            self.current_inj = self.N_inj_perStep - self.np_out
            self.low_row  = self.high_row
            self.high_row = self.high_row+self.current_inj
            self.total_inj=self.total_inj+self.current_inj
            print("cur inj: %d, low row %d, high row %d"%(self.current_inj,self.low_row,self.high_row))
            self.rangeParticleInj()
            ##self.printSys_dataframe()
            self.find_position(self.total_inj)
            self.printSys_dataframe()

        xx= self.system_particleP[0:self.total_inj,0]
        yy= self.system_particleP[0:self.total_inj,1]
        print("step: %d curent inj: %s n out: %s, total injected %s "%(self.counter,self.current_inj,self.np_out,self.total_inj))
        fileout.write("%d,%2.9f,%d\n" % (self.counter,self.dtime*self.counter,self.total_inj))
        fileout.flush()
        self.counter+=1
        ##current_state = self.state[:self.total_inj]
            
            
            #this does not ensure consecutive steady state
           ### if(self.current_inj == self.np_out):
           ###     self.steadycount+=1
           ###     if (self.steadycount == 10):
           ###         break
        fileout.close()
        return xx,yy


    #def run(self):
    #    self.step()




myParticle = ParticleSys(sys_row=10000000,sys_col=6,dtime=7.545e-8,total_part=10000000,nInjPs=1,sys_len=10,sys_wid=6)
#xx,yy =myParticle.step()
#print("check xx",xx)
#print("check yy",yy)
#print(myParticle.state['Y'])



fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)

def init():
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)
    return ln,

def update(frame):
    #ax.axvline(50, linestyle='--', color='black')
    xx,yy=myParticle.step()
    ln.set_data(xx,yy)
    ln.set_markersize(2)
    return ln,

#writer=animation.writers['ffmpeg']

ani = FuncAnimation(fig, update, frames=60,init_func=init, blit=True)
#ani.save('lines.mp4', writer=writer)
plt.plot((-3,-3),(-3,3))
plt.show()


