import numpy as np
import needle as ndl
import torch

def test_fft_1d():
    for input_size in [4, 6, 24, 62, 256]:
        x = np.random.randn(input_size).astype(np.float32)
        _x = ndl.Tensor(x)
        np.testing.assert_allclose(
            ndl.fft(_x).numpy(), 
            torch.fft.fft(torch.tensor(x)).numpy(), 
            atol=1e-5, rtol=1e-5
        )

def test_fft_2d():
    for batch_size in [6, 11]:
        for input_size in [4, 6, 24, 62, 256]:
            x = np.random.randn(batch_size, input_size).astype(np.float32)
            _x = ndl.Tensor(x)
            np.testing.assert_allclose(
                ndl.fft(_x).numpy(), 
                torch.fft.fft(torch.tensor(x)).numpy(), 
                atol=1e-5, rtol=1e-5
            )

def test_fft_3d():
    for batch_size in [6, 64]:
        for input_size in [4, 6, 24, 62, 256]:
            x = np.random.randn(batch_size, input_size, input_size).astype(np.float32)
            _x = ndl.Tensor(x)
            np.testing.assert_allclose(
                ndl.fft(_x).numpy(), 
                torch.fft.fft(torch.tensor(x)).numpy(), 
                atol=1e-5, rtol=1e-5
            )
            
            
def test_ifft_1d():
    for input_size in [4, 6, 24, 62, 256]:
        x = np.random.randn(input_size).astype(np.float32)
        _x = ndl.Tensor(x)
        forward = ndl.fft(_x)
        np.testing.assert_allclose(
            ndl.ifft(forward).numpy(),
            np.fft.ifft(torch.fft.fft(torch.tensor(x)).numpy()).astype(np.float32),
            atol=1e-5, rtol=1e-5
        )
        
def test_ifft_2d():
    for batch_size in [6, 11]:
        for input_size in [4, 6, 24, 62, 256]:
            x = np.random.randn(batch_size, input_size).astype(np.float32)
            _x = ndl.Tensor(x)
            forward = ndl.fft(_x)
            np.testing.assert_allclose(
                ndl.ifft(forward).numpy(),
                np.fft.ifft2(torch.fft.fft2(torch.tensor(x)).numpy()).astype(np.float32),
                atol=1e-5, rtol=1e-5
            )
            
def test_ifft_3d():
    para_1d = [4, 6, 24, 62, 256]  # Define the sizes you want to test
    para_3d = [6, 64]  # Define the batch sizes you want to test

    for batch_size in para_3d:
        for input_size in para_1d:
            x = np.random.randn(batch_size, input_size, input_size).astype(np.float32)
            _x = ndl.Tensor(x)
            forward = ndl.fft(_x)
            backward = ndl.ifft(forward)
            np.testing.assert_allclose(
                backward.numpy(), x, 
                atol=1e-5, rtol=1e-5
            )

            forward_dim = ndl.fft(_x, dim=1)
            backward_dim = ndl.ifft(forward_dim, dim=1)
            np.testing.assert_allclose(
                backward_dim.numpy(), x, 
                atol=1e-5, rtol=1e-5
            )


def test_rfft_1d():
    for input_size in [4, 6, 24, 62, 256]:
        x = np.random.randn(input_size).astype(np.float32)
        _x = ndl.Tensor(x)
        np.testing.assert_allclose(
            ndl.rfft(_x).numpy(),
            torch.fft.rfft(torch.tensor(x)).numpy(),
            atol=1e-5, rtol=1e-5
        )


def test_rfft_2d():
    for batch_size in [6, 11]:
        for input_size in [4, 6, 24, 62, 256]:
            x = np.random.randn(batch_size, input_size).astype(np.float32)
            _x = ndl.Tensor(x)
            np.testing.assert_allclose(
                ndl.rfft(_x, dim=(1,)).numpy(),
                torch.fft.rfftn(torch.tensor(x), dim=(1,)).numpy(),
                atol=1e-5, rtol=1e-5
            )
            

def test_rfft_3d():
    para_1d = [4, 6, 24, 32, 62, 64, 256]
    para_3d = [4, 6, 8, 32]

    for batch_size in para_3d:
        for input_size in para_1d:
            x = np.random.randn(batch_size, input_size, input_size).astype(np.float32)
            _x = ndl.Tensor(x)
            np.testing.assert_allclose(
                ndl.rfft(_x).numpy(), 
                torch.fft.rfftn(torch.tensor(x)).numpy(), 
                atol=1e-3, rtol=1e-4
            )
            np.testing.assert_allclose(
                ndl.rfft(_x, dim=(1,)).numpy(), 
                torch.fft.rfftn(torch.tensor(x), dim=(1,)).numpy(), 
                atol=1e-5, rtol=1e-5
            )


def test_irfft_1d():
    for input_size in [4, 6, 24, 62, 256]:
        x = np.random.randn(input_size).astype(np.float32)
        _x = ndl.Tensor(x)
        forward = ndl.rfft(_x)
        np.testing.assert_allclose(
            ndl.irfft(forward, [input_size]).numpy(),
            torch.fft.irfft(torch.fft.rfft(torch.tensor(x)), n=input_size).numpy(),
            atol=1e-5, rtol=1e-5
        )

def test_irfft_2d():
    for batch_size in [6, 11]:
        for input_size in [4, 6, 24, 62, 256]:
            x = np.random.randn(batch_size, input_size).astype(np.float32)
            _x = ndl.Tensor(x)
            forward = ndl.rfft(_x)
            original_shape = [batch_size, input_size]
            np.testing.assert_allclose(
                ndl.irfft(forward, original_shape).numpy(),
                torch.fft.irfft(torch.fft.rfft(torch.tensor(x)), n=input_size).numpy(),
                atol=1e-5, rtol=1e-5
            )

def test_irfft_3d():
    para_1d = [4, 6, 24, 62, 256]
    para_3d = [6, 64]
    for batch_size in para_3d:
        for input_size in para_1d:
            x = np.random.randn(batch_size, input_size, input_size).astype(np.float32)
            _x = ndl.Tensor(x)
            forward = ndl.rfft(_x)
            original_shape = [batch_size, input_size, input_size]
            np.testing.assert_allclose(
                ndl.irfft(forward, original_shape).numpy(),
                torch.fft.irfft(torch.fft.rfft(torch.tensor(x)), n=input_size).numpy(),
                atol=1e-5, rtol=1e-5
            )
