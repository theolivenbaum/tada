# TADA C# Porting Goal

The goal is to port the TADA generative framework for speech modeling from Python (PyTorch) to pure C# targeting .NET 10. The codebase will make use of modern C# features such as `Span<T>`, SIMD instructions (via `System.Numerics.Vectors` / intrinsics), and the `System.Numerics.Tensors` namespace to achieve good performance without taking dependencies on BLAS libraries.

The port focuses on text-to-speech functionality, including interacting with .wav files via the NAudio library. At this time, real-time microphone handling is out of scope.

## Project Structure

The target solution will consist of the following structure:
* **Tada**: A core class library containing the actual implementation of the model architecture, tokenization, arithmetic operations (SIMD/Tensor logic), and audio processing.
* **Tada.CLI**: A command-line application that allows users to perform text-to-speech tasks by interacting with the `Tada` core library, managing input and output to/from `.wav` files.

## Performance Considerations

* Usage of `Span<T>` and `Memory<T>` should be preferred to minimize allocations.
* Heavy numerical operations (like tensor dot products, softmax, layer normalization, etc.) should use hardware-accelerated SIMD intrinsics where applicable, falling back to scalar operations only if needed.
* Do **not** use BLAS or external C/C++ libraries.
