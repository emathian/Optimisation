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
	'''func 1 is the function from the first exercice.
	 func1 is define in R2 and convex.'''

	return  (x-y)**4 +2*x**2 +y**2 -x +2*y

def func2(x,y):
	'''func 2 is the function from the second exercice; 
	func2 is define in R2 and form a saddle.'''

	return  x**2 - y**2


def func3(x,y):
	'''func 3 is the function from the third exercice;
	 func3 is define in R2 and form of several kind of critic points.'''

	return  x**4 - x**3 - 20 * x**2 + x + 1 + y**4 - y**3 - 20 * y **2 + y + 1

def stop (cond, max_iteration,k,norm_grad,current_norm_grad, f_xk, f_xk1):
	''' stop is a function use both by the gradient descend method and 
	Newton method. We have fuour stop criteria such as :

	0 => a maximum number of iterations (k_max)
	1 => a minimal  norm to reach (g_min)
	2 => union of 0 and 1 conditions
	3 => at each step we assure that f(x_k+1)< f(x_k)

	This function return impossible if the first argument is different from this list.'''

	if cond ==0 :
		c_stop = k < max_iteration
	elif cond ==1  :
		c_stop = current_norm_grad > norm_grad
	elif cond==2 :
		c_stop =  k < max_iteration and current_norm_grad > norm_grad
	elif cond==3 :
		c_stop = f_xk1 < f_xk
			
	else:
		return 'impossible'	
	return c_stop		

def grad_f1(x0, y0):
	''' According to a couple of coordiantes, grad_f1 calculates the gradient 
	of func_1 and returns df/dx, df/dy and |g| (gradient's norm).
	NB : This norm is given according to the Euclidian definition.'''
	
	derivative_x0 = (4*(x0 - y0)**3+4*x0-1) # df/dx
	derivative_y0 = (-4*(x0-y0)**3+2*y0+2)  # df/dy
	norm_grad = sqrt(derivative_x0**2+derivative_y0**2) # |g|
	
	return [derivative_x0, derivative_y0, norm_grad]

def grad_f2(x, y):
	''' According to a couple of coordiantes, grad_f2 calculates the gradient 
	of func_2 and returns df/dx, df/dy and |g| (gradient's norm).'''

	derivative_x = 2 * x
	derivative_y = - 2* y
	norm_grad = sqrt(derivative_x**2+derivative_y**2)
	
	return [derivative_x, derivative_y,norm_grad]	

def grad_f3(x, y):
	''' According to a couple of coordiantes, grad_f3 calculates the gradient 
	of func_3 and returns df/dx, df/dy and |g| (gradient's norm).'''
	
	derivative_x = 4*x**3 - 3*x**2 -40*x + 1
	derivative_y = 4*y**3 - 3*y**2 - 40*y + 1
	norm_grad = sqrt(derivative_x**2+derivative_y**2)

	return [derivative_x, derivative_y,norm_grad]

def update_coordinates(x0,y0,a, function):
	''' The function update_coordinates is used by the function gradient_descent. 
	It allows according a couple  and a step 'a' to calculates a new couple of coordinates 
	(x_k+1 , y_k+1) acording the direction given by the result of grad_f% function.
	Update_coordinates is adapted to work with differentes functions (1,2,3), specified in third argument. '''

	if function == 1:
		d =grad_f1(x0,y0)

	elif function == 2:	
		d =grad_f2(x0,y0)
	elif function == 3:	
		d =grad_f3(x0,y0)		
	else :
		return ('NO function associetes')
	
	new_x = x0 + a*-1*d[0]
	new_y = y0 + a*-1*d[1]

	return[new_x,new_y]

def gradient_descent (x0,y0,cond_stop, a, max_iteration, norm_grad, f):
	''' gradient_descent gathered the previous function (func%, grad_f%, stop, and update_coordinates).
	According to a vector of initial conditions, a step 'a', and a stop criteria linked with a maximum 
	of iterations and a minimal norm, gradient_descend finds the minimum of the function (f) and returns
	lists of (x,y,z) calculated at each iteration, the number of iterations and the gradient's norm at the
	end of the process. '''

	# Initialization of coordiantes' lists
	x= [x0] 
	y= [y0]
	function = f
	if function == 1:
		z0 = func1(x0,y0)
		current_norm_grad = grad_f1(x0,y0)[2]
	elif function == 2:	
		z0 = func2(x0,y0)
		current_norm_grad = grad_f2(x0,y0)[2]
	elif function == 3:
		current_norm_grad = grad_f3(x0,y0)[2]
		z0 = func3(x0,y0)
	else :
		return ('NO function associetes')
	z=[z0]
	k =0 # k is the number of iteration at the begining
	z_1 = z0 + 1  # According to the stop criteria n°3, we artificially set f(x_k+1)< f(x_k), creating a f(x_k-1) point.

	c_stop = stop(cond_stop,max_iteration,k,norm_grad,current_norm_grad, z_1, z0)	
	while c_stop == True:
		if function == 1:
			new_xy =update_coordinates(x[-1],y[-1],a,1)
			x_new=new_xy[0]
			y_new=new_xy[1]
			z_new=func1(x_new,y_new)
			current_norm_grad = grad_f1(x[-1],y[-1])[2] # Calculated in order to the current norm
		elif function == 2:	
			new_xy =update_coordinates(x[-1],y[-1],a,2)
			x_new=new_xy[0]
			y_new=new_xy[1]
			z_new=func2(x_new,y_new)
			current_norm_grad = grad_f2(x[-1],y[-1])[2]

		elif function == 3:	
			new_xy =update_coordinates(x[-1],y[-1],a,3)
			x_new=new_xy[0]
			y_new=new_xy[1]
			z_new=func3(x_new,y_new)
			current_norm_grad = grad_f3(x[-1],y[-1])[2]	
		else :
			return ('NO function associetes')	

		# Memorization of results
		x.append(x_new)
		y.append(y_new)
		z.append(z_new)
		k+=1
		c_stop = stop(cond_stop,max_iteration,k,norm_grad,current_norm_grad, z[-2], z[-1]) # update stop criteria
		#print(current_norm_grad)
	
	return(x,y,z,current_norm_grad,k)			


def Hessian_f1(x,y):
	''' Hessian_f1 calculates and returns the hessian matrix of func1 
	according a couple of cordinates (x,y) '''

	H = np.zeros((2, 2))
	H[0][0] = 12*(x-y)**2+4 #d2f/dx2
	H[0][1] = -12*(x-y)**2 #d2f/dxdy
	H[1][0] = -12*(x-y)**2#d2f/dxdy
	H[1][1] =  12*(x-y)**2+2 #d2f/dy2
	return H

def Hessian_f2():
	''' Hessian_f2 calculates and returns the hessian matrix of func2 
	according a couple of cordinates (x,y) '''

	H = np.zeros((2, 2))
	H[0][0] = 2 #d2f/dx2
	H[0][1] = 0 #d2f/dxdy
	H[1][0] = 0#d2f/dxdy
	H[1][1] = -2#d2f/dy2
	return H


def Hessian_f3(x,y):
	''' Hessian_f3 calculates and returns the hessian matrix of func2 
	according a couple of cordinates (x,y) '''

	H = np.zeros((2, 2))
	H[0][0] = 12*x**2 -6*x -40 #d2f/dx2
	H[0][1] = 0 #d2f/dxdy
	H[1][0] = 0#d2f/dxdy
	H[1][1] =12*y**2 -6*y -40 #d2f/dy2
	
	return H


def Newton(x0,y0,cond_stop, max_iteration, norm_grad, f):
	'''Newton function gathered differantes the previous finctions : 
	-func% , grad_f%, Hessian_f% and stop
	According to a vector of initial vectors, a stop criteria associated with k_max and g_min
	Newton function returns lists of (x,y,z) calculated at each iteration, the number of iterations and the gradient's norm at the
	end of the process.
	Newton function at each coordinates calculates :
	-1)  d = ∆f/∆^2f such as d = H^-1. ∆f 
	-2) updinates coordinates suchs that x_k+1 = x_k + d (and the same for y_k+1)
	-3) updates z coordinates
	-4) memorize the coordiantes (x,y,z)
	-5) begin again until to check the stop criteria '''

	# Initialization of lists
	x= [x0]
	y= [y0]
	v_grad = np.zeros((2, 1))
	if f == 1:
		z0 = func1(x0,y0)
		# calculation of ∆f to initialize the norm when k=0
		v_grad[0][0]= -1*grad_f1(x[-1],y[-1])[0] 
		v_grad[1][0]=-1*grad_f1(x[-1],y[-1])[1]
		# calculation of d = ∆f/∆^2f to initialize the norm when k=0
		d =  np.dot(np.linalg.inv(Hessian_f1(x[-1],y[-1])),v_grad)
	elif f==2 :
		z0 = func2(x0,y0)
		v_grad[0][0]= -1*grad_f2(x[-1],y[-1])[0]
		v_grad[1][0]=-1*grad_f2(x[-1],y[-1])[1]
		d =  np.dot(np.linalg.inv(Hessian_f2()),v_grad)

	elif f==3 :
		z0 = func3(x0,y0)
		v_grad[0][0]= -1*grad_f3(x[-1],y[-1])[0]
		v_grad[1][0]=-1*grad_f3(x[-1],y[-1])[1]
		d =  np.dot(np.linalg.inv(Hessian_f3(x[-1], y[-1])),v_grad)	

	else :
		return ('NO function associeted')	

	z=[z0]
	z_1 = z0 +1 # According to the stop criteria n°3, we artificially set f(x_k+1)< f(x_k), creating a f(x_k-1) point.
	k =0 # k is the number of iterations
	current_norm_grad = sqrt(d[0][0]**2+d[1][0]**2) # initialisation of the current norm of d

	c_stop = stop(cond_stop,max_iteration,k,norm_grad,current_norm_grad, z_1, z0)	
	
	while c_stop == True:
		if f==1 :
			# for the first function we check if the Hessian matrix is defined positive.
			H_eigen = np.linalg.eigvals(Hessian_f1(x[-1],y[-1])) 
			if H_eigen[0]>0 and H_eigen[1] > 0:
				v_grad = np.zeros((2, 1))
				v_grad[0][0]= -1*grad_f1(x[-1],y[-1])[0]
				v_grad[1][0]=-1*grad_f1(x[-1],y[-1])[1]
				d = np.dot(np.linalg.inv(Hessian_f1(x[-1],y[-1])),v_grad)
		elif f==2 :
	
			v_grad = np.zeros((2, 1))
			v_grad[0][0]= -1*grad_f2(x[-1],y[-1])[0]
			v_grad[1][0]=-1*grad_f2(x[-1],y[-1])[1]
			d = np.dot(np.linalg.inv(Hessian_f2()),v_grad)

		elif f==3 :
			H_eigen = np.linalg.eigvals(Hessian_f1(x[-1],y[-1]))

			if H_eigen[0]>0 and H_eigen[1] > 0:
				v_grad = np.zeros((2, 1))
				v_grad[0][0]= -1*grad_f3(x[-1],y[-1])[0]
				v_grad[1][0]=-1*grad_f3(x[-1],y[-1])[1]
				d = np.dot(np.linalg.inv(Hessian_f3(x[-1], y[-1])),v_grad)
		else :
			return('No function')		

		current_norm_grad = sqrt(d[0][0]**2+d[1][0]**2) # Updating of the current norm of d
		# Updating new coordiantes
		x_new=x[-1] + d[0][0]
		y_new=y[-1] + d[1][0]
		if f==1:
			z_new=func1(x_new,y_new)
		elif f==2:
			z_new=func2(x_new,y_new)
		elif f==3:
			z_new=func3(x_new,y_new)	
		else :
			return('No function')	
		# Memorization of results	
		x.append(x_new)
		y.append(y_new)
		z.append(z_new)
		
		k+=1
		c_stop = stop(cond_stop,max_iteration,k,norm_grad,current_norm_grad, z[-2], z[-1])	 # updates stop conditions
	
	# Erasing of the last result if the stop criteria choosen is '3'
	if cond_stop == 3:
	 	x.pop(len(x)-1)	
	 	y.pop(len(y)-1)	
	 	z.pop(len(z)-1)	
	return(x,y,z,current_norm_grad,k)		
	


############################################################################################
#											MAIN  :       								   #	
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
	sol_N1 =gradient_descent(1,1,0,0.009,1000,0.01,1)
	print('At the end (x,y,z) are equal to  :', sol_N1[0][-1], sol_N1[1][-1], sol_N1[2][-1])
	print('The gradient norm is   :', sol_N1[3])
	print('The number of iteration is:', sol_N1[4])
	print('1.2) If the number of iterations is determining we set k to 50 ')
	sol_N2 =gradient_descent(1,1,0,0.009,50,0.01,1)
	print('At the end (x,y,z) are equal to  :', sol_N2[0][-1], sol_N2[1][-1], sol_N2[2][-1])
	print('The gradient norm is   :', sol_N2[3])
	print('The number of iteration is:', sol_N2[4])
	print('')
	print('2) If the gradient norm is determining and we set a maximal norm to |g| = 0.01 (with k=1000) \n')
	sol_N3 =gradient_descent(1,1,1,0.009,1000,0.01,1)
	print('At the end (x,y,z) are equal to  :', sol_N3[0][-1], sol_N3[1][-1], sol_N3[2][-1])
	print('The gradient norm is   :', sol_N3[3])
	print('The number of iteration is:', sol_N3[4])
	print('')
	print('3) If the gradient norm and the number of iteration are determining and we set a maximal norm to |g| = 0.01 and k=1000 \n')
	sol_N4 =gradient_descent(1,1,2,0.009,1000,0.01,1)
	print('At the end (x,y,z) are equal to  :' ,sol_N4[0][-1], sol_N4[1][-1], sol_N4[2][-1])
	print('The gradient norm is   :' ,sol_N4[3])
	print('The number of iteration is:' , sol_N4[4])
	print('')

if Which_question == 4 :

	sol =gradient_descent(1,1,2,0.009,1000,0.01,1)
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
	sol =gradient_descent(1,1,2,0.009,1000,0.01,1)
	print('At the end (x,y,z) are equal to  :', sol[0][-1], sol[1][-1], sol[2][-1])
	print('The gradient norm is   :', sol_N1[3])
	print('The number of iteration is:', sol_N1[4])

if Which_question == 6 :
	Which_subquestion = int(input("Which sub_question ?  \n 1 = inital conditions \n 2 = stopping criteria \n 3 = gradient's norm \n "))
	if Which_subquestion == 1 :
		sol_N =gradient_descent(1,1,2,0.09,10000,0.01,1)
		sol_N2 =gradient_descent(-0.5,1,2,0.09,10000,0.01,1)
		sol_N3 =gradient_descent(-1,0.5,2,0.009,10000,0.01,1)
		sol_N4 =gradient_descent(-1,-1,2,0.0009,10000,0.01,1)
		sol_N5 =gradient_descent(0.5,-1,2,0.09,10000,0.01,1)
		

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
		sol_N1 =gradient_descent(0.7,1,0,0.09,1,0.01,1)
		sol_N2 =gradient_descent(0.8,1,0,0.09,3,0.01,1)
		sol_N3 =gradient_descent(0.9,1,0,0.09,5,0.01,1)
		sol_N4 =gradient_descent(1,1,0,0.09,10,0.01,1)

		
		# cond norm grad
		sol_N5 =gradient_descent(0.4,-1,1,0.09,10000,5,1)
		sol_N6 =gradient_descent(0.5,-1,1,0.09,10000,1,1)
		sol_N7 =gradient_descent(0.6,-1,1,0.09,10000,0.001,1)


		# Two cond
		sol_N8 =gradient_descent(-0.5,-0.75,2,0.09,100,0.1,1)
		sol_N9 =gradient_descent(-0.8,-0.75,2,0.09,1000,0.0001,1)
		sol_N10 =gradient_descent(-1.1,-0.75,2,0.09,10000,0.00001,1)

		# # Cond 3 
		sol_N11 =gradient_descent(-0.25,0.75,3,0.09,10000,0.01,1)
		sol_N12 =gradient_descent(-0.3,0.75,3,0.09,10000,0.01,1)
		sol_N13 =gradient_descent(-0.15,0.75,3,0.09,10000,0.01,1)
		

		x = sol_N1[0] 
		y = sol_N1[1] #defines the y variable
		z = sol_N1[2]

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

		x10 = sol_N10[0] #defines the x variable
		y10 =sol_N10[1] #defines the y variable
		z10=sol_N10[2]

		x11 = sol_N11[0] #defines the x variable
		y11 =sol_N11[1] #defines the y variable
		z11= sol_N11[2]


		x12 = sol_N12[0] #defines the x variable
		y12 =sol_N12[1] #defines the y variable
		z12=sol_N12[2]

		x13 = sol_N13[0] #defines the x variable
		y13 =sol_N13[1] #defines the y variable
		z13=sol_N13[2]


		print('Norm N11 :', sol_N11[3], 'Nb_iter :', sol_N11[4])
		print('Norm N12 :', sol_N12[3], 'Nb_iter :', sol_N12[4])
		print('Norm N13 :', sol_N13[3], 'Nb_iter :', sol_N13[4])
		print('Norm N12 x, y ,z :', sol_N12[0][-1],  sol_N12[1][-1],  sol_N12[2][-1])
		fig = plt.figure() #opens a figure environment
		ax = fig.gca(projection='3d') #to perform a 3D plot
		X = np.arange(-1, 1, 0.1) #x rangedjq<d
		Y = np.arange(-1, 1, 0.1) #y range
		X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
		Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
		ax.plot(x, y, z, label='k=1', color='#01153e') #Navy 
		ax.plot(x2, y2, z2, label='k=3', color='#601ef9')
		ax.plot(x3, y3, z3, label='k=5', color='#0165fc')  #plot definition and options 
		ax.plot(x4, y4, z4, label='k=10', color='#d0fefe')  #plot definition and options 
		ax.plot(x5, y5, z5, label='|g|=5', color='#f10c45')
		ax.plot(x6, y6, z6, label='|g|=1', color='#d46a7e')
		ax.plot(x7, y7, z7, label='|g|=0.001,', color='#cb416b')
		ax.plot(x8, y8, z8, label='k=100  |g|=0.1', color='#b0ff9d')
		ax.plot(x9, y9, z9, label='k=1000 |g|=1*10^-4', color='#76cd26')
		ax.plot(x10, y10, z10, label='k=1*10^4 |g|=1*10^-5', color='#cdfd02')

		ax.plot(x11, y11, z11, label='f(x_k+1) < f(x-k)', color='tan')
		ax.plot(x12, y12, z12, label='f(x_k+1) < f(x-k)', color='gold')
		ax.plot(x13, y13, z13, label='f(x_k+1) < f(x-k)', color='wheat')

		ax.legend() #adds a legend
		mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()

	if Which_subquestion == 3 :
				# cond norm grad
		sol_N1=gradient_descent(0.8,1,2,0.01,100,0.01,1)
		sol_N2 =gradient_descent(0.9,1,2,0.005,100,0.01,1)
		sol_N3 =gradient_descent(1,1,2,0.0005,100,0.01,1)


		print('Nb iter N1' , sol_N1[4])
		print('Nb iter N2' , sol_N2[4])
		print('Nb iter N3' , sol_N3[4])

		x = sol_N1[0] 
		y = sol_N1[1] #defines the y variable
		z = sol_N1[2]

		x2 = sol_N2[0] #defines the x variable
		y2 =sol_N2[1] #defines the y variable
		z2=sol_N2[2]

		x3 = sol_N3[0] #defines the x variable
		y3 =sol_N3[1] #defines the y variable
		z3=sol_N3[2]

		fig = plt.figure() #opens a figure environment
		ax = fig.gca(projection='3d') #to perform a 3D plot
		X = np.arange(-1, 1, 0.1) #x rangedjq<d
		Y = np.arange(-1, 1, 0.1) #y range
		X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
		Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
		ax.plot(x, y, z, label='a = 1e-2', color='#01153e') #Navy 
		ax.plot(x2, y2, z2, label='a =5e-3', color='#601ef9')
		ax.plot(x3, y3, z3, label='a = 5e-4', color='#d0fefe')  #plot definition and options 
		

		ax.legend() #adds a legend
		mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()

if Which_question == 7 :
	print( 'Calcultation of Hessian matrix: \n [[12*(x-y)**2+4 , -12*(x-y)**2] \n [-12*(x-y)**2, 12*(x-y)**2+2]]\n')

if Which_question == 8 :

	print("Implementation of Newton method includes : \n -func1 \n -stop \n -gradf1 \n -Hessianf1 \n -Newton")

if Which_question == 9 :

	sol =Newton(1,1,2,1000,0.01)
	x = sol[0] #defines the x variable
	y =sol[1] #defines the y variable
	z=sol[2]

	fig = plt.figure() #opens a figure environment
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

if Which_question == 10 :
	sol =Newton(1,1,2,1000,0.01,1)
	print('At the end (x,y,z) are equal to  :', sol[0][-1], sol[1][-1], sol[2][-1])
	print('The gradient norm is   :', sol[3])
	print('The number of iteration is:', sol[4])

if Which_question==11 :

	Which_subquestion = int(input("Which sub_question ?  \n 1 = inital conditions \n 2 = stopping criteria \n 3 = gradient's norm \n "))
	if Which_subquestion == 1 :
		sol_N =Newton(1,1,2,10000,0.01,1)
		sol_N2 =Newton(-0.5,1,2,10000,0.01,1)
		sol_N3 =Newton(-1,0.5,2,10000,0.01,1)
		sol_N4 =Newton(-1,-1,2,10000,0.01,1)
		sol_N5 =Newton(0.5,-1,2,10000,0.01,1)
		

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

		print(sol_N[4],sol_N2[4],sol_N3[4],sol_N4[4],sol_N5[4])

		fig = plt.figure() #opens a figure environment
		ax = fig.gca(projection='3d') #to perform a 3D plot
		X = np.arange(-1, 1, 0.1) #x rangedjq<d
		Y = np.arange(-1, 1, 0.1) #y range
		X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
		Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
		ax.plot(x, y, z, label='(x,y)=(1,1)', color='#01153e')
		ax.plot(x2, y2, z2, label='(x,y)=(-0.5,1) '  , color='#d0fefe')
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
		sol_N1 =Newton(0.8,1,0,2,0.01,1)
		sol_N2 =Newton(0.9,1,0,3,0.01,1)
		sol_N3 =Newton(1,1,0,5,0.01,1)
		sol_N4 =Newton(1.1,1,0,10,0.01,1)

		# print(sol_N1[4],sol_N2[4],sol_N3[4],sol_N4[4]) 
		# print(sol_N1[3],sol_N2[3],sol_N3[3],sol_N4[3]) 
		# # cond norm grad
		sol_N5 =Newton(0.4,-1,1,10000,1*10**-12,1)
		sol_N6 =Newton(0.5,-1,1,10000,0.001,1)
		sol_N7 =Newton(0.6,-1,1,10000,0.01,1)

		# print(sol_N5[4],sol_N6[4],sol_N7[4])
		# # Two cond
		sol_N8 =Newton(-0.6,-0.75,2,10,0.1,1)
		sol_N9 =Newton(-0.8,-0.75,2,100000,1*10**-14,1)
		sol_N10 =Newton(-1,-0.75,2,10000,1*10**-20,1)
		print(sol_N8[4],sol_N9[4],sol_N10[4])
		print(sol_N8[3],sol_N9[3],sol_N10[3])
		# # Cond 3 
		sol_N11 =Newton(-0.25,0.75,3,10000,0.01,1)
		sol_N12 =Newton(-0.3,0.75,3,10000,0.01,1)
		sol_N13 =Newton(-0.15,0.75,3,10000,0.01,1)

		x = sol_N1[0] 
		y = sol_N1[1] #defines the y variable
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

		x10 = sol_N10[0] #defines the x variable
		y10 =sol_N10[1] #defines the y variable
		z10=sol_N10[2]

		x11 = sol_N11[0] #defines the x variable
		y11 =sol_N11[1] #defines the y variable
		z11= sol_N11[2]

	

		x12 = sol_N12[0] #defines the x variable
		y12 =sol_N12[1] #defines the y variable
		z12=sol_N12[2]

		x13 = sol_N13[0] #defines the x variable
		y13 =sol_N13[1] #defines the y variable
		z13=sol_N13[2]

		fig = plt.figure() #opens a figure environment
		ax = fig.gca(projection='3d') #to perform a 3D plot
		X = np.arange(-1, 1, 0.1) #x rangedjq<d
		Y = np.arange(-1, 1, 0.1) #y range
		X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
		Z= (X-Y)**4 +2*X**2 +Y**2 -X +2*Y #defines the function values

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
		ax.plot(x, y, z, label='k=2', color='#01153e') #Navy 
		ax.plot(x2, y2, z2, label='k=3', color='#601ef9')
		ax.plot(x3, y3, z3, label='k=5', color='#0165fc')  #plot definition and options 
		ax.plot(x4, y4, z4, label='k=10', color='#d0fefe')  #plot definition and options 
		ax.plot(x5, y5, z5, label='|g|=1e-12', color='#f10c45')
		ax.plot(x6, y6, z6, label='|g|=1e-3', color='#d46a7e')
		ax.plot(x7, y7, z7, label='|g|=1e-2', color='#ffcfdc')
		ax.plot(x8, y8, z8, label='k=10  |g|=1e-1', color='#154406')
		ax.plot(x9, y9, z9, label='k=1*10^5 |g|=1e-12', color='#76cd26')
		ax.plot(x10, y10, z10, label='k=1*10^4 |g|=1e-20', color='#cdfd02')
		ax.plot(x11, y11, z11, label='f(x_k+1) < f(x-k)', color='tan')
		ax.plot(x12, y12, z12, label='f(x_k+1) < f(x-k)', color='gold')
		ax.plot(x13, y13, z13, label='f(x_k+1) < f(x-k)', color='wheat')
		ax.legend() #adds a legend
		mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()	




if Which_question==12 :


	sol_N1 =Newton(0.7,1,2,500,0.01,1)
	sol_gd1 =gradient_descent(0.7,1,2,0.09,500,0.01,1)
	

	
	sol_gd4 =gradient_descent(1,1,2,0.01,5,0.01,1)
	sol_N4 =Newton(1,1,2,5,0.01,1)

	# cond norm grad
	sol_N5 =Newton(0.4,-1,2,10000,5,1)
	sol_gd5 =gradient_descent(0.4,-1,2,0.09,10000,5,1)

	sol_N7 =Newton(0.6,-1,2,10000,0.01,1)
	sol_gd7 =gradient_descent(0.6,-1,2,0.09,10000,0.01,1)

	# Two cond
	
	sol_N9 =Newton(-0.8,-0.75,2,100000,0.000000001,1)
	sol_gd9 =gradient_descent(-0.8,-0.75,2,0.0009,1000,0.000000001,1)



	print('Norm sol N1 ', sol_N1[3])
	print('Iter sol 1 ', sol_N1[4])
	print('x sol 1 ', sol_N1[0][-1],'y sol 1 ', sol_N1[1][-1], 'z sol 1 ', sol_N1[2][-1], )


	print('Norm sol gd1 ', sol_gd1[3])
	print('Iter sol 1 ', sol_gd1[4])
	print('x sol 1 ', sol_gd1[0][-1],'y sol 1 ', sol_gd1[1][-1], 'z sol 1 ', sol_gd1[2][-1], )


	print('Norm sol N4 ', sol_N4[3])
	print('Iter sol 1 ', sol_N4[4])
	print('x sol 1 ', sol_N4[0][-1],'y sol 1 ', sol_N4[1][-1], 'z sol 1 ', sol_N4[2][-1], )

	print('Norm sol gd4 ', sol_gd4[3])
	print('Iter sol 1 ', sol_gd4[4])
	print('x sol 1 ', sol_gd4[0][-1],'y sol 1 ', sol_gd4[1][-1], 'z sol 1 ', sol_gd4[2][-1], )


	print('Norm sol N5 ', sol_N5[3])
	print('Iter sol 1 ', sol_N5[4])
	print('x sol 1 ', sol_N5[0][-1],'y sol 1 ', sol_N5[1][-1], 'z sol 1 ', sol_N5[2][-1], )

	print('Norm sol gd5 ', sol_gd5[3])
	print('Iter sol 1 ', sol_gd5[4])
	print('x sol 1 ', sol_gd5[0][-1],'y sol 1 ', sol_gd5[1][-1], 'z sol 1 ', sol_gd5[2][-1], )


	print('Norm sol N7 ', sol_N7[3])
	print('Iter sol 1 ', sol_N7[4])
	print('x sol 1 ', sol_N7[0][-1],'y sol 1 ', sol_N7[1][-1], 'z sol 1 ', sol_N7[2][-1], )

	print('Norm sol gd7 ', sol_gd7[3])
	print('Iter sol 1 ', sol_gd7[4])
	print('x sol 1 ', sol_gd7[0][-1],'y sol 1 ', sol_gd7[1][-1], 'z sol 1 ', sol_gd7[2][-1], )
	
	print('Norm sol N9 ', sol_N9[3])
	print('Iter sol 1 ', sol_N9[4])
	print('x sol 1 ', sol_N9[0][-1],'y sol 1 ', sol_N9[1][-1], 'z sol 1 ', sol_N9[2][-1], )

	print('Norm sol gd9 ', sol_gd9[3])
	print('Iter sol 1 ', sol_gd9[4])
	print('x sol 1 ', sol_gd9[0][-1],'y sol 1 ', sol_gd9[1][-1], 'z sol 1 ', sol_gd9[2][-1], )


if Which_question==13 :

	fig = plt.figure() #opens a figure environment
	ax = fig.gca(projection='3d') #to perform a 3D plot
	X = np.arange(-2, 2, 0.1) #x rangedjq<d
	Y = np.arange(-2, 2, 0.1) #y range
	X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
	Z= X**2 - Y**2  #defines the function values

	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.legend() #adds a legend
	#mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()


if Which_question==14 :

	sol_1 =gradient_descent(0,0,0,1,40,0.01,2) # Condition stop 2
	sol_2 =gradient_descent(-5,5,2,0.01,40,0.01,2)


	print('Norm sol 1 :', sol_1[3])
	print('Nb iteration :', sol_1[4])

	print('Norm sol 2 :', sol_2[0][-1], sol_2[1][-1], sol_2[2][-1])
	x = sol_1[0] 
	y = sol_1[1] #defines the y variable
	z = sol_1[2]

	x2 = sol_2[0] #defines the x variable
	y2 =sol_2[1] #defines the y variable
	z2=sol_2[2]	

	fig = plt.figure() #opens a figure environment
	ax = fig.gca(projection='3d') #to perform a 3D plot
	X = np.arange(-6, 6, 0.1) #x rangedjq<d
	Y = np.arange(-6, 6, 0.1) #y range
	X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
	Z= X**2 - Y**2  #defines the function values

	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.plot(x, y, z, label='(x0, y0)= (0,0)', color='y') #Navy 
	ax.plot(x2, y2, z2, label='(x0, y0)= (-5,5)', color='pink')
	ax.legend() #adds a legend
	mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()



if Which_question==15 :	

	sol_1 =Newton(0,0,2,40,0.01,2) # Condition stop 2
	sol_2 =Newton(1,1,2,40,0.01,2)
	print(sol_2, sol_2[1][-1])
	print(sol_2[0][-1], sol_2[1][-1])
	x = sol_1[0] 
	y = sol_1[1] #defines the y variable
	z = sol_1[2]

	x2 = sol_2[0] #defines the x variable
	y2 =sol_2[1] #defines the y variable
	z2=sol_2[2]	

	fig = plt.figure() #opens a figure environment
	ax = fig.gca(projection='3d') #to perform a 3D plot
	X = np.arange(-6, 6, 0.1) #x rangedjq<d
	Y = np.arange(-6, 6, 0.1) #y range
	X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
	Z= X**2 - Y**2  #defines the function values

	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.plot(x, y, z, label='(x0, y0)= (0,0)', color='y') #Navy 
	ax.plot(x2, y2, z2, label='(x0, y0)= (-5,5)', color='pink')
	ax.legend() #adds a legend
	mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()


if Which_question ==16 :
	print ('Voir l explication')


if Which_question == 17 :


	fig = plt.figure() #opens a figure environment
	ax = fig.gca(projection='3d') #to perform a 3D plot
	X = np.arange(-6, 6, 0.1) #x rangedjq<d
	Y = np.arange(-6, 6, 0.1) #y range
	X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
	Z=  X**4 - X**3 - 20 * X**2 + X + 1 + Y**4 - Y**3 - 20 * Y **2 + Y + 1
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.legend() #adds a legend
	#mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()

if Which_question == 18 :

	sol_1 =gradient_descent(3,4,2,0.01,10000,0.01,3) # Condition stop 2
	sol_2 =gradient_descent(-3,-3,2,0.01,10000,0.01,3)
	sol_3 =gradient_descent(-4,-3,2,0.01,10000,0.01,3)

	print('Norm sol 1:',sol_1[3])
	print('Nb iter sol1: ',sol_1[4])
	print('coordonnees sol1: ',sol_1[0][-1] ,sol_1[1][-1], sol_1[2][-1]  )

	print('Norm sol 2:',sol_2[3])
	print('Nb iter sol 2: ',sol_2[4])
	print('coordonnees sol 2: ',sol_2[0][-1] ,sol_2[1][-1], sol_2[2][-1]  )

	print('Norm sol 3:', sol_3[3])
	print('Nb iter sol 3 : ', sol_3[4])
	print('coordonnees sol 3: ',sol_3[0][-1] ,sol_3[1][-1], sol_3[2][-1]  )


	x = sol_1[0] 
	y = sol_1[1] #defines the y variable
	z = sol_1[2]

	x2 = sol_2[0] #defines the x variable
	y2 =sol_2[1] #defines the y variable
	z2=sol_2[2]	

	x3 = sol_3[0] #defines the x variable
	y3 =sol_3[1] #defines the y variable
	z3=sol_3[2]	

	fig = plt.figure() #opens a figure environment
	ax = fig.gca(projection='3d') #to perform a 3D plot
	X = np.arange(-4.5, 4.5, 0.1) #x rangedjq<d
	Y = np.arange(-4.5, 4.5, 0.1) #y range
	X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
	Z=  X**4 - X**3 - 20 * X**2 + X + 1 + Y**4 - Y**3 - 20 * Y **2 + Y + 1
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.plot(x, y, z, label='(x0, y0)= (3,4)', color='tan') #Navy 
	ax.plot(x2, y2, z2, label='(x0, y0)= (-3,-3)', color='gold')
	ax.plot(x3, y3, z3, label='(x0, y0)= (-4,-3)', color='wheat')
	ax.legend() #adds a legend
	#mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()



if Which_question == 19 :
	sol_1 =Newton(3,4,2,10000,0.01,3) # Condition stop 2
	sol_2 =Newton(-3,-3,2,10000,0.01,3)
	sol_3 =Newton(-4,-3,2,10000,0.01,3)

	print('Norm sol 1:',sol_1[3])
	print('Nb iter sol1: ',sol_1[4])
	print('coordonnees sol1: ',sol_1[0][-1] ,sol_1[1][-1], sol_1[2][-1]  )

	print('Norm sol 2:',sol_2[3])
	print('Nb iter sol 2: ',sol_2[4])
	print('coordonnees sol 2: ',sol_2[0][-1] ,sol_2[1][-1], sol_2[2][-1]  )

	print('Norm sol 3:', sol_3[3])
	print('Nb iter sol 3 : ', sol_3[4])
	print('coordonnees sol 3: ',sol_3[0][-1] ,sol_3[1][-1], sol_3[2][-1]  )


	x = sol_1[0] 
	y = sol_1[1] #defines the y variable
	z = sol_1[2]

	x2 = sol_2[0] #defines the x variable
	y2 =sol_2[1] #defines the y variable
	z2=sol_2[2]	

	x3 = sol_3[0] #defines the x variable
	y3 =sol_3[1] #defines the y variable
	z3=sol_3[2]	

	fig = plt.figure() #opens a figure environment
	ax = fig.gca(projection='3d') #to perform a 3D plot
	X = np.arange(-4.5, 4.5, 0.1) #x rangedjq<d
	Y = np.arange(-4.5, 4.5, 0.1) #y range
	X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
	Z=  X**4 - X**3 - 20 * X**2 + X + 1 + Y**4 - Y**3 - 20 * Y **2 + Y + 1
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.plot(x, y, z, label='(x0, y0)= (3,4)', color='tan') #Navy 
	ax.plot(x2, y2, z2, label='(x0, y0)= (-3,-3)', color='gold')
	ax.plot(x3, y3, z3, label='(x0, y0)= (-4,-3)', color='wheat')
	ax.legend() #adds a legend
	#mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()


if Which_question == 20 :
	Which_subquestion = int(input("Which sub_question ?  \n 1 = critical condition with descend gradient method \n 2 = 1 = critical condition with Newton method  "))
	if Which_subquestion == 1 :
		sol_1 =gradient_descent(0.5,0.5,2,0.01,10000,0.01,3) # Condition stop 2
		sol_2 =gradient_descent(0,3,2,0.01,10000,0.01,3)
		sol_3 =gradient_descent(-3,0,2,0.01,10000,0.01,3)

		print('Norm sol 1:',sol_1[3])
		print('Nb iter sol1: ',sol_1[4])
		print('coordonnees sol1: ',sol_1[0][-1] ,sol_1[1][-1], sol_1[2][-1]  )

		print('Norm sol 2:',sol_2[3])
		print('Nb iter sol 2: ',sol_2[4])
		print('coordonnees sol 2: ',sol_2[0][-1] ,sol_2[1][-1], sol_2[2][-1]  )

		print('Norm sol 3:', sol_3[3])
		print('Nb iter sol 3 : ', sol_3[4])
		print('coordonnees sol 3: ',sol_3[0][-1] ,sol_3[1][-1], sol_3[2][-1]  )


		x = sol_1[0] 
		y = sol_1[1] #defines the y variable
		z = sol_1[2]

		x2 = sol_2[0] #defines the x variable
		y2 =sol_2[1] #defines the y variable
		z2=sol_2[2]	

		x3 = sol_3[0] #defines the x variable
		y3 =sol_3[1] #defines the y variable
		z3=sol_3[2]	

		fig = plt.figure() #opens a figure environment
		ax = fig.gca(projection='3d') #to perform a 3D plot
		X = np.arange(-4.5, 4.5, 0.1) #x rangedjq<d
		Y = np.arange(-4.5, 4.5, 0.1) #y range
		X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
		Z=  X**4 - X**3 - 20 * X**2 + X + 1 + Y**4 - Y**3 - 20 * Y **2 + Y + 1
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
		ax.plot(x, y, z, label='(x0, y0)= (0.5 , 0.5)', color='tan') #Navy 
		ax.plot(x2, y2, z2, label='(x0, y0)= (0,3)', color='gold')
		ax.plot(x3, y3, z3, label='(x0, y0)= (-3,0)', color='wheat')
		ax.legend() #adds a legend
		#mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()
	if Which_subquestion == 2:
		sol_1 =Newton(0.5,0.5,2,10000,0.01,3) # Condition stop 2
		sol_2 =Newton(0,3,2,10000,0.01,3)
		sol_3 =Newton(-3,0,2,10000,0.01,3)

		print('Norm sol 1:',sol_1[3])
		print('Nb iter sol1: ',sol_1[4])
		print('coordonnees sol1: ',sol_1[0][-1] ,sol_1[1][-1], sol_1[2][-1]  )

		print('Norm sol 2:',sol_2[3])
		print('Nb iter sol 2: ',sol_2[4])
		print('coordonnees sol 2: ',sol_2[0][-1] ,sol_2[1][-1], sol_2[2][-1]  )

		print('Norm sol 3:', sol_3[3])
		print('Nb iter sol 3 : ', sol_3[4])
		print('coordonnees sol 3: ',sol_3[0][-1] ,sol_3[1][-1], sol_3[2][-1]  )


		x = sol_1[0] 
		y = sol_1[1] #defines the y variable
		z = sol_1[2]

		x2 = sol_2[0] #defines the x variable
		y2 =sol_2[1] #defines the y variable
		z2=sol_2[2]	

		x3 = sol_3[0] #defines the x variable
		y3 =sol_3[1] #defines the y variable
		z3=sol_3[2]	

		fig = plt.figure() #opens a figure environment
		ax = fig.gca(projection='3d') #to perform a 3D plot
		X = np.arange(-4.5, 4.5, 0.1) #x rangedjq<d
		Y = np.arange(-4.5, 4.5, 0.1) #y range
		X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
		Z=  X**4 - X**3 - 20 * X**2 + X + 1 + Y**4 - Y**3 - 20 * Y **2 + Y + 1
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
		ax.plot(x, y, z, label='(x0, y0)= (0.5 , 0.5)', color='tan') #Navy 
		ax.plot(x2, y2, z2, label='(x0, y0)= (0,3)', color='gold')
		ax.plot(x3, y3, z3, label='(x0, y0)= (-3,0)', color='wheat')
		ax.legend() #adds a legend
		#mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()		



