GPU based calculations for approximating a 'multinomial like' distribution where each draw has different probabilities accross the classes.

Turned out for the cases I was interested in (around 10 classes) that this was only marginally faster than the CPU, when the equivalent calculations on the CPU used doubles, not floats as this does. It might become useful if the number of classes were larger, or in applications in which more work is done on the GPU. It never got past the prototype and hence has no tests or documentation... use at your own risk!