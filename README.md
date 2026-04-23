# Digit Recognition: A neural network written in c++ trained on the mnist dataset

Recognises digits at a rate of 95.6%. 
Multithreading and other optimizations helped speed up training by 35x.

# Building and Running

The project can be built using `cmake`.
To run the project, do: 

```bash
unzip mnist.zip
mkdir build && cd build
cmake ..
make
./DigitRecognition
```
