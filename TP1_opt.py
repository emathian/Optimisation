#Imports from the matplotlib library
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import sqrt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import decimal



def func1(x,y):
	return  (x-y)**4 +2*x**2 +y**2 -x +2*y

def stop (cond, max_iteration,k,norm_grad,current_norm_grad):
	if cond ==0 :
		c_stop = k < max_iteration
	elif cond ==1  :
		c_stop = current_norm_grad > norm_grad
	elif cond==2 :
		c_stop =  k < max_iteration and current_norm_grad > norm_grad
	else:
		return 'impossible'	
	return c_stop		

def grad_f1(x0, y0):
	# According to the manuel gradient calculation
	derivative_x0 = (4*(x0 - y0)**3+4*x0-1)
	derivative_y0 = (-4*(x0-y0)**3+2*y0+2)
	norm_grad = sqrt(derivative_x0**2+derivative_y0**2)
	#print(norm_grad)
	return [derivative_x0, derivative_y0,norm_grad]

def update_coordinates(x0,y0,a):
	d =grad_f1(x0,y0)
	new_x = x0 - a*d[0]
	new_y = y0 - a*d[1]
	return[new_x,new_y]

def gradient_descent (x0,y0,cond_stop, a, max_iteration, norm_grad ):
	# stop 0 : stop condition is max iterations
	x= [x0]
	y= [y0]
	z0 = func1(x0,y0)
	z=[z0]
	k =0 # k is the number of iteration
	current_norm_grad = grad_f1(x0,y0)[2]
	c_stop = stop(cond_stop,max_iteration,k,norm_grad,current_norm_grad)	
	while c_stop == True:
		new_xy =update_coordinates(x[-1],y[-1],a)
		x_new=new_xy[0]
		y_new=new_xy[1]
		z_new=func1(x_new,y_new)
		current_norm_grad = grad_f1(x[-1],y[-1])[2]
		x.append(x_new)
		y.append(y_new)
		z.append(z_new)
		k+=1
		c_stop = stop(cond_stop,max_iteration,k,norm_grad,current_norm_grad)
		#print(current_norm_grad)
	
	return(x,y,z,current_norm_grad,k)			


def Hessian_f1(x,y):
	H = np.zeros((2, 2))
	H[0][0] = 12*(x-y)**2+4 #d2f/dx2
	H[0][1] = -12*(x-y)**2 #d2f/dxdy
	H[1][0] = -12*(x-y)**2#d2f/dxdy
	H[1][1] =  12*(x-y)**2+2 #d2f/dy2
	return H

def Newton(x0,y0,cond_stop, max_iteration, norm_grad):
	x= [x0]
	y= [y0]
	z0 = func1(x0,y0)
	z=[z0]
	k =0 # k is the number of iteration
	v_grad = np.zeros((2, 1))
	v_grad[0][0]= -1*grad_f1(x[-1],y[-1])[0]
	v_grad[1][0]=-1*grad_f1(x[-1],y[-1])[1]
	
	d =  np.dot(np.linalg.inv(Hessian_f1(x[-1],y[-1])),v_grad)
	
	current_norm_grad = sqrt(d[0][0]**2+d[1][0]**2)
	c_stop = stop(cond_stop,max_iteration,k,norm_grad,current_norm_grad)	
	
	while k<max_iteration:
		H_eigen = np.linalg.eigvals(Hessian_f1(x[-1],y[-1])) 
		if H_eigen[0]>0 and H_eigen[1] > 0:
			v_grad = np.zeros((2, 1))
			v_grad[0][0]= -1*grad_f1(x[-1],y[-1])[0]
			v_grad[1][0]=-1*grad_f1(x[-1],y[-1])[1]
			
			d = np.dot(np.linalg.inv(Hessian_f1(x[-1],y[-1])),v_grad)
			current_norm_grad = sqrt(d[0][0]**2+d[1][0]**2)
			x_new=x[-1] + d[0][0]
			y_new=y[-1] + d[1][0]
			z_new=func1(x_new,y_new)
			x.append(x_new)
			y.append(y_new)
			z.append(z_new)
			c_stop = stop(cond_stop,max_iteration,k,norm_grad,current_norm_grad)	
			k+=1
			
	return(x,y,z,current_norm_grad,k)		
	
############################################################################################

Which_question = int(input("Which question ?    "))

if Which_question == 1 :
	fig = plt.figure() #opens a figure environment
	fig.suptitle("function f(x,y)", fontsize=16)
	ax = fig.gca(projection='3d') #to perform a 3D plot
	X = np.arange(-1, 1, 0.01) #x rangedjq<d
	Y = np.arange(-1, 1, 0.01) #y range
	X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
	Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.legend() #adds a legend
	#mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()

if Which_question == 2 :
	print('The gradient of f function writted |g| is  :\n derivative_x0 = (4*(x0 - y0)**3+4*x0-1) \n derivative_y0 = (-4*(x0-y0)**3+2*y0+2)' )

if Which_question == 3 :
	print('Effect of the stopping criteria : \n We fix initial conditions to (1,1), the convergence rate  a=0.09 \n 1.1) If the number of iterations is determining we set k to 1000 \n')
	sol_N1 =gradient_descent(1,1,0,0.009,1000,0.01)
	print('At the end (x,y,z) are equal to  :', sol_N1[0][-1], sol_N1[1][-1], sol_N1[2][-1])
	print('The gradient norm is   :', sol_N1[3])
	print('The number of iteration is:', sol_N1[4])
	print('1.2) If the number of iterations is determining we set k to 50 ')
	sol_N2 =gradient_descent(1,1,0,0.009,50,0.01)
	print('At the end (x,y,z) are equal to  :', sol_N2[0][-1], sol_N2[1][-1], sol_N2[2][-1])
	print('The gradient norm is   :', sol_N2[3])
	print('The number of iteration is:', sol_N2[4])
	print('')
	print('2) If the gradient norm is determining and we set a maximal norm to |g| = 0.01 (with k=1000) \n')
	sol_N3 =gradient_descent(1,1,1,0.009,1000,0.01)
	print('At the end (x,y,z) are equal to  :', sol_N3[0][-1], sol_N3[1][-1], sol_N3[2][-1])
	print('The gradient norm is   :', sol_N3[3])
	print('The number of iteration is:', sol_N3[4])
	print('')
	print('3) If the gradient norm and the number of iteration are determining and we set a maximal norm to |g| = 0.01 and k=1000 \n')
	sol_N4 =gradient_descent(1,1,2,0.009,1000,0.01)
	print('At the end (x,y,z) are equal to  :' ,sol_N4[0][-1], sol_N4[1][-1], sol_N4[2][-1])
	print('The gradient norm is   :' ,sol_N4[3])
	print('The number of iteration is:' , sol_N4[4])
	print('')

if Which_question == 4 :

	sol =gradient_descent(1,1,2,0.009,1000,0.01)
	x = sol[0] #defines the x variable
	y =sol[1] #defines the y variable
	z=sol[2]

	fig = plt.figure() #opens a figure environment
	fig.suptitle("function f(x,y) and optimization solution with (x0,y0)=(1,1), a=0.09 |g|=0.01 and cond_stop=2", fontsize=12)
	ax = fig.gca(projection='3d') #to perform a 3D plot
	X = np.arange(-1, 1, 0.01) #x rangedjq<d
	Y = np.arange(-1, 1, 0.01) #y range
	X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
	Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.plot(x, y, z, label='learning rate', color='#01153e')
	ax.legend() #adds a legend
	mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()

if Which_question == 5 :
	sol =gradient_descent(1,1,2,0.009,1000,0.01)
	print('At the end (x,y,z) are equal to  :', sol[0][-1], sol[1][-1], sol[2][-1])
	print('The gradient norm is   :', sol_N1[3])
	print('The number of iteration is:', sol_N1[4])

if Which_question == 6 :
	Which_subquestion = int(input("Which sub_question ?  \n 1 = inital conditions \n 2 = stopping criteria \n 3 = gradient's norm \n "))
	if Which_subquestion == 1 :
		sol_N =gradient_descent(1,1,2,0.09,10000,0.01)
		sol_N2 =gradient_descent(-0.5,1,2,0.09,10000,0.01)
		sol_N3 =gradient_descent(-1,0.5,2,0.009,10000,0.01)
		sol_N4 =gradient_descent(-1,-1,2,0.0009,10000,0.01)
		sol_N5 =gradient_descent(0.5,-1,2,0.09,10000,0.01)
		

		x = sol_N[0] #defines the x variable
		y =sol_N[1] #defines the y variable
		z=sol_N[2]

		x2 = sol_N2[0] #defines the x variable
		y2 =sol_N2[1] #defines the y variable
		z2=sol_N2[2]

		x3 = sol_N3[0] #defines the x variable
		y3 =sol_N3[1] #defines the y variable
		z3=sol_N3[2]


		x4 = sol_N4[0] #defines the x variable
		y4 =sol_N4[1] #defines the y variable
		z4=sol_N4[2]



		x5 = sol_N5[0] #defines the x variable
		y5 =sol_N5[1] #defines the y variable
		z5=sol_N5[2]

		fig = plt.figure() #opens a figure environment
		ax = fig.gca(projection='3d') #to perform a 3D plot
		X = np.arange(-1, 1, 0.01) #x rangedjq<d
		Y = np.arange(-1, 1, 0.01) #y range
		X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
		Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
		ax.plot(x, y, z, label='(x,y)=(1,1)', color='#01153e')
		ax.plot(x2, y2, z2, label='(x,y)=(-0.5,1)', color='#d0fefe')
		ax.plot(x3, y3, z3, label='(x,y)=(-1,0.5)', color='#0165fc')  #plot definition and options 
		ax.plot(x4, y4, z4, label='(x,y)=(-1,-1)', color='#cdfd02')  #plot definition and options 
		ax.plot(x5, y5, z5, label='(x,y)=(0.5,-1)', color='#f10c45')
		ax.legend() #adds a legend
		mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()

		if Which_subquestion == 2 :
			# cond nb iter	
			sol_N1 =gradient_descent(1,1,0,0.09,5000,0.01)
			sol_N2 =gradient_descent(1,1,0,0.09,500,0.01)
			sol_N3 =gradient_descent(1,1,0,0.09,100,0.01)
			sol_N4 =gradient_descent(1,1,0,0.09,50,0.01)

			# cond norm grad
			sol_N5 =gradient_descent(0.6,-1,1,0.09,10000,2)
			sol_N6 =gradient_descent(0.5,-1,1,0.09,10000,1)
			sol_N7 =gradient_descent(0.6,-1,1,0.09,10000,0.01)


			# Two cond
			sol_N8 =gradient_descent(-0.9,0.5,2,0.0009,100000,0.1)
			sol_N9 =gradient_descent(-1,0.5,2,0.0009,1000000,0.1)
			sol_N10 =gradient_descent(-0.8,0.5,2,0.0009,100000000,0.01)


			x = sol_N1[0] 
			y = sol_N[11] #defines the y variable
			z = sol_N1[2]

			x2 = sol_N2[0] #defines the x variable
			y2 =sol_N2[1] #defines the y variable
			z2=sol_N2[2]

			x3 = sol_N4[0] #defines the x variable
			y3 =sol_N4[1] #defines the y variable
			z3=sol_N4[2]


			x4 = sol_N4[0] #defines the x variable
			y4 =sol_N4[1] #defines the y variable
			z4=sol_N4[2]



			x5 = sol_N5[0] #defines the x variable
			y5 =sol_N5[1] #defines the y variable
			z5=sol_N5[2]


			x6 = sol_N6[0] #defines the x variable
			y6 =sol_N6[1] #defines the y variable
			z6=sol_N6[2]

			x7 = sol_N7[0] #defines the x variable
			y7 =sol_N7[1] #defines the y variable
			z7=sol_N7[2]


			x8 = sol_N8[0] #defines the x variable
			y8 =sol_N8[1] #defines the y variable
			z8=sol_N8[2]


			x9 = sol_N9[0] #defines the x variable
			y9 =sol_N9[1] #defines the y variable
			z9=sol_N9[2]

			x10 = sol_N20[0] #defines the x variable
			y10 =sol_N20[1] #defines the y variable
			z10=sol_N20[2]


			fig = plt.figure() #opens a figure environment
			ax = fig.gca(projection='3d') #to perform a 3D plot
			X = np.arange(-1, 1, 0.01) #x rangedjq<d
			Y = np.arange(-1, 1, 0.01) #y range
			X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
			Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

			surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
			ax.plot(x, y, z, label='k=5000', color='#01153e') #Navy 

			ax.plot(x2, y2, z2, label='k=500', color='#601ef9')
			ax.plot(x3, y3, z3, label='k=100', color='#0165fc')  #plot definition and options 
			ax.plot(x4, y4, z4, label='k=50', color='#d0fefe')  #plot definition and options 
			ax.plot(x5, y5, z5, label='|g|=2', color='#f10c45')
			ax.plot(x6, y6, z6, label='|g|=1', color='#d46a7e')
			ax.plot(x7, y7, z7, label='|g|=0.01', color='#ffcfdc')
			ax.plot(x8, y8, z8, label='k=1*10^5  |g|=0.1', color='#154406')
			ax.plot(x9, y9, z9, label='k=1*10^6 |g|=0.1', color='#76cd26')
			ax.plot(x10, y10, z10, label='k=1*10^8 |g|=0.01', color='#cdfd02')

			ax.legend() #adds a legend
			mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')
			plt.show()
'''
 var  k
sol_N2 =gradient_descent(1,1,0,0.0009,5000,0.01)
sol_N2 =gradient_descent(1,1,0,0.0009,500,0.01)
sol_N4 =gradient_descent(1,1,0,0.0009,100,0.01)
sol_N4 =gradient_descent(1,1,0,0.0009,50,0.01)

 cond norm grad
sol_N5 =gradient_descent(1.5,-1,1,0.0000009,1000000,2)
sol_N6 =gradient_descent(1.4,-1,1,0.0000009,1000000,1)
sol_N7 =gradient_descent(1.3,-1,1,0.0000009,1000000,0.01)


 Two cond
sol_N8 =gradient_descent(-0.9,0.5,2,0.00000009,100000,0.1)
sol_N9 =gradient_descent(-1,0.5,2,0.00000009,1000000,0.1)
sol_N20 =gradient_descent(-0.8,0.5,2,0.00000009,100000000,0.01)

SOlution en fct du point initiale
sol =gradient_descent(1,-2,2,0.09,10000,0.001)
sol_N2 =gradient_descent(1.5,-1,2,0.0000009,1000000,0.1)
sol_N =gradient_descent(1,1,2,0.00009,10000,0.001)
sol_N4 =gradient_descent(-1,0.5,2,0.00000009,100000000,0.1)
sol_N4 =gradient_descent(-1,-1,2,0.0009,10000,0.01)
sol_N5 =gradient_descent(0.5,-1,2,0.0000009,10000000,0.01)

sol_N = gradient_descent(-0.5,1,2,0.09,200,0.0001)


print(sol_N[0][-1],sol_N[1][-1])
print(sol_N[3])
sol_z = (sol[0]-sol[1])**4+2*sol[0]**2+sol[1]**2-sol[0]+2*sol[1]


Definition of what to plot

fig = plt.figure() #opens a figure environment
ax = fig.gca(projection='3d') #to perform a 3D plot
X = np.arange(-1, 1, 0.01) #x rangedjq<d
Y = np.arange(-1, 1, 0.01) #y range
X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options

x = sol_N[0] #defines the x variable
print(x[0:10])
y = sol_N[1] #defines the y variable
print(y[0:10])
z = sol_N[2]
print(z[0:10])


print(sol_N[3], 'norm_grad')
print(sol_N[4], 'nb iteration')




x2 = sol_N2[0] #defines the x variable
y2 =sol_N2[1] #defines the y variable
z2=sol_N2[2]

x3 = sol_N4[0] #defines the x variable
y3 =sol_N4[1] #defines the y variable
z3=sol_N4[2]


x4 = sol_N4[0] #defines the x variable
y4 =sol_N4[1] #defines the y variable
z4=sol_N4[2]



x5 = sol_N5[0] #defines the x variable
y5 =sol_N5[1] #defines the y variable
z5=sol_N5[2]


x6 = sol_N6[0] #defines the x variable
y6 =sol_N6[1] #defines the y variable
z6=sol_N6[2]

x7 = sol_N7[0] #defines the x variable
y7 =sol_N7[1] #defines the y variable
z7=sol_N7[2]


x8 = sol_N8[0] #defines the x variable
y8 =sol_N8[1] #defines the y variable
z8=sol_N8[2]


x9 = sol_N9[0] #defines the x variable
y9 =sol_N9[1] #defines the y variable
z9=sol_N9[2]

x10 = sol_N20[0] #defines the x variable
y10 =sol_N20[1] #defines the y variable
z10=sol_N20[2]



print(sol_N8[3], 'norm_grad sol N8')
print(sol_N8[4], 'iter sol N8')

print(sol_N9[3], 'norm_grad sol N9')
print(sol_N9[4], 'iter sol N9')

print(sol_N20[3], 'norm_grad sol N10')
print(sol_N20[4], 'iter sol N10')


ax.plot(x, y, z, label='k=5000', color='#01153e') #Navy 

ax.plot(x2, y2, z2, label='k=500', color='#601ef9')
ax.plot(x3, y3, z3, label='k=100', color='#0165fc')  #plot definition and options 
ax.plot(x4, y4, z4, label='k=50', color='#d0fefe')  #plot definition and options 
ax.plot(x5, y5, z5, label='|g|=2', color='#f10c45')
ax.plot(x6, y6, z6, label='|g|=1', color='#d46a7e')
ax.plot(x7, y7, z7, label='|g|=0.01', color='#ffcfdc')
ax.plot(x8, y8, z8, label='k=1*10^5  |g|=0.1', color='#154406')
ax.plot(x9, y9, z9, label='k=1*10^6 |g|=0.1', color='#76cd26')
ax.plot(x10, y10, z10, label='k=1*10^8 |g|=0.01', color='#cdfd02')

ax.legend() #adds a legend
mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

Runs the plot command
plt.show()
'''
