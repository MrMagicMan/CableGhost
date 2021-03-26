#This is to simulate a 3 cable driven ghost

import math
import matplotlib.pyplot as plt
import numpy
from numpy import sqrt, dot, cross                       
from numpy.linalg import norm  
import yaml


#To calculate the point of the ghost [xg,yg] we will use the other points known locations and the distance of the cable to figure it out


class motorElement:
    def __init__(self,x,y,z,cableLength):
        self.P = numpy.array([x,y,z]);
        self.cableLength = cableLength;

#def convertToSpherical(element):
#    rho = element.P[0]^2+element.P[1]^2 + element.P[2]^2
#    theta = atan2(element.P[1]^2,element.P[0]^2)
#    phi =


    
def create_sphere(cx,cy,cz, r, resolution=360):
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = numpy.linspace(0, 2*numpy.pi, 2*resolution)
    theta = numpy.linspace(0, numpy.pi, resolution)

    theta, phi = numpy.meshgrid(theta, phi)

    r_xy = r*numpy.sin(theta)
    x = cx + numpy.cos(phi) * r_xy
    y = cy + numpy.sin(phi) * r_xy
    z = cz + r * numpy.cos(theta)

    return numpy.stack([x,y,z])


def calcDistanceBetweenPoints(P0,P1):
    distance = sqrt((P1[0]-P0[0])**2 + (P1[1]-P0[1])**2 + (P1[2]-P0[2])**2)
    return distance

# Find the intersection of three spheres                 
# P1,P2,P3 are the centers, r1,r2,r3 are the radii       
# Implementaton based on Wikipedia Trilateration article.                              
def trilaterate(P1,P2,P3,r1,r2,r3):                      
    temp1 = P2-P1                                        
    e_x = temp1/norm(temp1)                              
    temp2 = P3-P1                                        
    i = dot(e_x,temp2)                                   
    temp3 = temp2 - i*e_x                                
    e_y = temp3/norm(temp3)                              
    e_z = cross(e_x,e_y)                                 
    d = norm(P2-P1)                                      
    j = dot(e_y,temp2)                                   
    x = (r1*r1 - r2*r2 + d*d) / (2*d)                    
    y = (r1*r1 - r3*r3 -2*i*x + i*i + j*j) / (2*j)       
    temp4 = r1*r1 - x*x - y*y                            
    if temp4<0:                                          
        raise Exception("The three spheres do not intersect!");
    z = sqrt(temp4)                                      
    p_12_a = P1 + x*e_x + y*e_y + z*e_z                  
    p_12_b = P1 + x*e_x + y*e_y - z*e_z                  
    return p_12_a,p_12_b         


#starting with the forward kinimatics (calculate Pg based on P1,P2,P3 and the distances..
devices = []
devices.append(motorElement(0,0,30,0))
devices.append(motorElement(-60,50,70,0))
devices.append(motorElement(60,70,60,0))





Pdesire = numpy.array([0,10,10])
print("Desired Point is ",Pdesire)

#Figure out lengths



for obj in devices:
    
    obj.cableLength = calcDistanceBetweenPoints(obj.P,Pdesire)





#convert for ouput
P0 = devices[0].P;
P1 = devices[1].P;
P2 = devices[2].P;

PA = numpy.array([P0, P1, P2])

checkoutput = trilaterate(P0,P1,P2,devices[0].cableLength,devices[1].cableLength,devices[2].cableLength)

if checkoutput[0][2] < checkoutput[1][2]:
    output=checkoutput[0]
else:
    output=checkoutput[1]

        
    

for obj in devices:
    print(obj.P,obj.cableLength)


print("Output Calculation is ", output);


# draw sphere
#u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#x = np.cos(u)*np.sin(v)
#y = np.sin(u)*np.sin(v)
#z = np.cos(v)
#ax.plot_wireframe(x, y, z, color="r")


sphere1 = create_sphere(devices[0].P[0],devices[0].P[1],devices[0].P[2], devices[0].cableLength, resolution=10)
sphere2 = create_sphere(devices[1].P[0],devices[1].P[1],devices[1].P[2], devices[1].cableLength, resolution=10)
sphere3 = create_sphere(devices[2].P[0],devices[2].P[1],devices[2].P[2], devices[2].cableLength, resolution=10)


#plot the intersection
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(PA.transpose()[0], PA.transpose()[1], PA.transpose()[2],linewidth=3, cmap='r');
ax.scatter3D(output[0], output[1], output[2],linewidth=50, marker='+');

ax.plot_wireframe(sphere1[0], sphere1[1], sphere1[2], alpha=0.1 ,color="r")
ax.plot_wireframe(sphere2[0], sphere2[1], sphere2[2], alpha=0.1 ,color="g")
ax.plot_wireframe(sphere3[0], sphere3[1], sphere3[2], alpha=0.1 ,color="b" )

#plt.plot(PA.transpose()[0],PA.transpose()[1],PA.transpose()[2],'ro')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(60, 35)

plt.show()
    
              
