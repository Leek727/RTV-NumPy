import numpy as np
import cv2
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

class RTV:
    def __init__(self, left, lambda_S=0.01, sigma=3, sigma_annealing_end=0.6):
        self.left = left.astype(np.float32)
        self.lambda_S = lambda_S
        self.sigma = sigma
        self.sigma_annealing_end = sigma_annealing_end

        if np.max(self.left) > 1.0:
            self.left /= 255.0

    def apply_Lis(self, x, Ux, Vx, Uy, Vy):
        """
        Given a HxW matrix x, HxW matricies Ux, Vx, Uy, Vy, returns L_Is x (a HxW matrix)
        """
        x = x.astype(np.float32)
        dx = np.diff(x, axis=1, append=0)
        dy = np.diff(x, axis=0, append=0)

        RHSx = Ux * Vx * dx
        RHSy = Uy * Vy * dy

        Lsx = -np.diff(RHSx, axis=1, prepend=0) # since -Gx_prepend = G^T x_append
        Lsy = -np.diff(RHSy, axis=0, prepend=0)

        return Lsx + Lsy
    
    def get_UV(self, Is, sigma):
        """
        Given structure image Is, computes Ux, Vx, Uy, Vy and returns as
        HW vectors
        Eqn 8 and 9
        """
        Is = Is.astype(np.float32)
        H, W, C = Is.shape
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
        # epsilon values from RTV paper
        epsilon = 1e-3
        epsilon_s = 2e-2

        # compute U
        # np.diff appends a 0 to the right side, thus the last forward difference calulation
        # is just the last value and the shape of the matrix is preserved
        dx = np.diff(Is, axis=1, append=0) # (H, W, 3)
        dy = np.diff(Is, axis=0, append=0)

        # calculate per channel Ux,Uy matricies then take max
        Ux = np.zeros((H, W))
        Uy = np.zeros((H, W))
        Gdx = cv2.GaussianBlur(dx, (kernel_size, kernel_size), sigmaX=sigma)
        Gdy = cv2.GaussianBlur(dy, (kernel_size, kernel_size), sigmaX=sigma)

        Recipx = 1 / (np.max(np.abs(Gdx), axis=2) + epsilon)
        Recipy = 1 / (np.max(np.abs(Gdy), axis=2) + epsilon)

        Ux = cv2.GaussianBlur(Recipx, (kernel_size, kernel_size), sigmaX=sigma)
        Uy = cv2.GaussianBlur(Recipy, (kernel_size, kernel_size), sigmaX=sigma)

        # compute V
        Vx = 1 / (np.mean(abs(dx), axis=2) + epsilon_s)
        Vy = 1 / (np.mean(abs(dy), axis=2) + epsilon_s)

        return Ux, Vx, Uy, Vy

    def solve_system(self):
        H, W, C = self.left.shape
        Ux, Vx, Uy, Vy = 0, 0, 0, 0

        iters = 3
        sigma = self.sigma
        annealing_rate = (self.sigma - self.sigma_annealing_end) / (iters - 1)
        

        S = self.left.copy()
        for i in range(iters):
            Ux, Vx, Uy, Vy = self.get_UV(S, sigma)
            for channel in range(C):
                b = self.left[:, :, channel].reshape(H*W)
                def mv(x):
                    x = x.reshape(H, W)
                    mat_res = x + self.lambda_S * self.apply_Lis(x, Ux, Vx, Uy, Vy)
                    return mat_res.reshape(H*W)
            
                A = LinearOperator((H*W, H*W), matvec=mv)

                x, exit_code = cg(A, b, x0=S[:, :, channel].reshape(H * W), rtol=1e-6)
                S[:, :, channel] = x.reshape((H,W))
            
            sigma -= annealing_rate
            
        return np.clip(S * 255, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="bicycle.png")
    parser.add_argument("--lambda_S", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=3)
    parser.add_argument("--sigma_annealing_end", type=float, default=0.6)
    parser.add_argument("--save_path", type=str, default="out.png")
    args = parser.parse_args()
    
    left = cv2.imread(args.image)
    rtv = RTV(left, args.lambda_S, args.sigma, args.sigma_annealing_end)

    s = rtv.solve_system()
    cv2.imwrite(args.save_path, s)