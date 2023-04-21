# AI_Project-Reproducing-Image-Using-GA
Reproducing image using Genetic Algorithm in Artificial Intelligent

## Project Steps
#### The steps to follow in order to reproduce an image are as follows:

- Read an image
- Prepare the fitness function
- Create an instance of the pygad.GA class with the appropriate parameters
- Run PyGAD
- Plot results
- Calculate some statistics

### Read an Image
There is an image named fruit.jpg in this project to be reproduced using Genetic Algorithm.
 
```
# Reading target image to be reproduced using Genetic Algorithm 
target_im = imageio.v2.imread('/content/fruit.jpg')
target_im = numpy.asarray(target_im/255, dtype=numpy.float)
```
```
# Target image after enconding
target_chromosome = img2chromosome(target_im)
```

![fruit](https://user-images.githubusercontent.com/128599179/233535313-22fe07a2-3c24-48c5-9d7c-31c886716de2.jpg)

Based on the chromosome representation used in the example, the pixel values can be either in the 0-255, 0-1, or any other ranges. the range of pixel values affect other parameters like the range from which the random values are selected during mutation and also the range of the values used in the initial population. So, be consistent.

## Prepare the Fitness Function
A fitness function for calculating the fitness value for each solution in the population. This function must be a maximization function that accepts 2 parameters representing a solution and its index. It returns a value representing the fitness value.

The fitness value is calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes. The img2chromosome() function is called before the fitness function to represent the image as a vector because the genetic algorithm can work with 1D chromosomes.


```
def img2chromosome(img_arr):
    return numpy.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))
```

```
def chromosome2img(vector, shape):
    # Check if the vector can be reshaped according to the specified shape.
    if len(vector) != functools.reduce(operator.mul, shape):
        raise ValueError("A vector of length {vector_length} into an array of shape {shape}."
        .format(vector_length=len(vector), shape=shape))
    return numpy.reshape(a=vector, newshape=shape)
```

```
def fitness_fun(ga_instance, solution, solution_idx):
    fitness = numpy.sum(numpy.abs(target_chromosome-solution))
    # Negating the fitness value to make it increasing rather than decreasing.
    fitness = numpy.sum(target_chromosome) - fitness
    return fitness
```    

```
def callback(ga_instance):
    print("Generation = {gen}".format(gen=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    if ga_instance.generations_completed % 500 == 0:
        matplotlib.pyplot.imsave('solution_'+str(ga_instance.generations_completed)+'.png', 
                                 chromosome2img(ga_instance.best_solution()[0], target_im.shape))
```

### Create an Instance of the pygad.GA Class
It is very important to use random mutation and set the mutation_by_replacement to True. Based on the range of pixel values, the values assigned to the init_range_low, init_range_high, random_mutation_min_val, and random_mutation_max_val parameters should be changed.

Here we are using the image pixel values range from 0 to 1, then set init_range_low and random_mutation_min_val to 0 as they are but change init_range_high and random_mutation_max_val to 1.

```
import pygad

ga_instance = pygad.GA(num_generations=50000,
                       num_parents_mating=10,
                       fitness_func=fitness_fun,
                       sol_per_pop=20,
                       num_genes=target_im.size,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=0.01,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0)
                       
```
### Run PyGAD
Simply, call the run() method to run PyGAD.


```
ga_instance.run()
```

### Plot Results
After the run() method completes, the fitness values of all generations can be viewed in a plot using the plot_result() method.

```
ga_instance.plot_result()
```

### Calculate Some Statistics
After this Calculation the actual generated image of fruit.jpg for given number of generation will be shown.

```
# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
```

```
if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} 

generations.".format(best_solution_generation=ga_instance.best_solution_generation))
result = chromosome2img(solution, target_im.shape)
matplotlib.pyplot.imshow(result)
matplotlib.pyplot.title("PyGAD & GARI for Reproducing Images")
matplotlib.pyplot.show()
```
<img align="center" width="250" height="200" src="https://user-images.githubusercontent.com/128599179/233591086-dda66fc2-bd6d-4867-8873-f12284abd5a5.png">

## Generations

#### 10 Generation
<img align="center" width="250" height="200" src="https://user-images.githubusercontent.com/128599179/233593960-797afb96-87fe-4f9c-9352-ccb8bf2e1b57.png">

#### 100 Generation
<img align="center" width="250" height="200" src="https://user-images.githubusercontent.com/128599179/233594654-a1b466d1-c5af-44b3-a107-5715e28f9b28.png">

#### 2000 Generation
<img align="center" width="250" height="200" src="https://user-images.githubusercontent.com/128599179/233594731-4f1e7e0e-1568-4883-bc60-e6dffac8f4b8.png">

#### 5000 Generation
<img align="center" width="250" height="200" src="https://user-images.githubusercontent.com/128599179/233594797-cd3f730f-aabf-4845-9ec2-ef56441fd9ea.png">

#### 10000 Generation
<img align="center" width="250" height="200" src="https://user-images.githubusercontent.com/128599179/233594871-7ae59916-f37e-4adf-b84e-ae3ade56c7c5.png">

#### 20000 Generation
<img align="center" width="250" height="200" src="https://user-images.githubusercontent.com/128599179/233594919-e7da3f8a-98c8-421e-b487-22f1c2c6c7e8.png">

#### 50000 Generation
<img align="center" width="250" height="200" src="https://user-images.githubusercontent.com/128599179/233591086-dda66fc2-bd6d-4867-8873-f12284abd5a5.png">

