import numpy as np

#defining constants
a0 = 4.124
a11 = -2.097*1e3
a12 = 7.97*1e3
a111 = 1.294*1e4
a112 = -1.950*1e4
a123 = -2.5*1e4
t0 = 115

#delta G function
def delta_G(px,py,pz,Tc):
    temp = 0.5*a0*(Tc - t0)*(px**2 + py**2 +pz**2)
    temp += 0.25*a11*(px**4 + py**4 + pz**4)
    temp += 0.5*a12*(px**2*py**2 + pz**2*py**2 + px**2*pz**2)
    temp += (a12*(px**6 + py**6 + pz**6))/6
    temp += 0.5*a112*(px**4*(pz**2 + py**2) + py**4*(px**2 + pz**2) + pz**4*(px**2 + py**2))
    temp += 0.5*a123*(px**2*py**2*pz**2)
    return temp

#neglect the trivial solution
#del (delta G)/del px
def f1(px,py,pz,Tc):
    temp = a0*(Tc - t0)
    temp += a11*px**2 + a12*(py**2 + pz**2) + a111*px**4 + a112*2*(px*(py**2 + pz**2)) + a112*(py**4 + pz**4) + a123*(py**2*pz**2)
    return list(temp)[0]

#del(delat G)/del py
def f2(px,py,pz,Tc):
    temp = a0*(Tc - t0)
    temp += a11*py**2 + a12*(px**2 + pz**2) + a111*py**4 + a112*2*(py*(px**2 + pz**2)) + a112*(px**4 + pz**4) + a123*(px**2*pz**2)
    return list(temp)[0]

#del(delat G)/del pz
def f3(px,py,pz,Tc):
    temp = a0*(Tc - t0)
    temp += a11*pz**2 + a12*(px**2 + py**2) + a111*pz**4 + a112*2*(pz*(px**2 + py**2)) + a112*(px**4 + py**4) + a123*(px**2*py**2)
    return list(temp)[0]

#derivatives of f1
def f1_px(px,py,pz):
    return list(2*a11*px + 4*a111*px**3 + a112*2*(pz**2 + py**2))[0]

def f1_py(px,py,pz):
    return list(2*a12*py + 4*a112*px*py + 4*a112*py**3 + 2*a123*py*pz**2)[0]

def f1_pz(px,py,pz):
    return list(2*a12*pz + 4*a112*px*pz + 4*a112*pz**3 + 2*a123*pz*py**2)[0]

#derivatives of f2
def f2_px(px,py,pz):
    return list(2*a12*px + 4*a112*px*py + 4*a112*px**3 + 2*a123*px*pz**2)[0]

def f2_py(px,py,pz):
    return list(2*a11*py + 4*a111*py**3 + a112*2*(px**2 + pz**2))[0]

def f2_pz(px,py,pz):
    return list(2*a12*pz + 4*a112*pz*py + 4*a112*pz**3 + 2*a123*pz*px**2)[0]

#derivatives of f3
def f3_pz(px,py,pz):
    return list(2*a11*pz + 4*a111*pz**3 + a112*2*(px**2 + py**2))[0]

def f3_px(px,py,pz):
    return list(2*a12*px + 4*a112*px*pz + 4*a112*px**3 + 2*a123*px*py**2)[0]

def f3_py(px,py,pz):
    return list(2*a12*py + 4*a112*py*pz + 4*a112*py**3 + 2*a123*py*px**2)[0]

#f = [f1_px,f1_py,f1_pz,f2_px,f2_py,f2_pz,f3_px,f3_py,f3_pz]

#Jacobian matrix
def J(px,py,pz):
    temp = [[f1_px(px,py,pz),f1_py(px,py,pz),f1_pz(px,py,pz)],[f2_px(px,py,pz),f2_py(px,py,pz),f2_pz(px,py,pz)],[f3_px(px,py,pz),f3_py(px,py,pz),f3_pz(px,py,pz)]]
    #for k in range(len(f)):
        #print(f[k](px,py,pz))
    return np.array(temp)

#functional
def F(px,py,pz,Tc):
    return np.array([[f1(px,py,pz,Tc)],[f2(px,py,pz,Tc)],[f3(px,py,pz,Tc)]])

if __name__ == '__main__':
    Tc = np.arange(80,110,1)
    for i in range(len(Tc)):
        x = np.array([[10],[10],[10]])
        itr = 100
        for j in range(itr):
            #j1 = J(x[0],x[1],x[2])
            #print(np.shape(j1))
            x = x - np.linalg.inv(J(x[0],x[1],x[2]))@F(x[0],x[1],x[2],Tc[i])
        print(f'Delta G = {delta_G(x[0],x[1],x[2],Tc[i])} for Tc = {Tc[i]}')
        
