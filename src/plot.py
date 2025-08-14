import numpy as np
import matplotlib.pyplot as plt

# ---- パラメータ設定 ----
A = 10                # Rastrigin関数の定数
range_val = 5.12      # 定義域 [-range_val, range_val]
step = 0.1            # 分解能

# ---- 格子点生成 ----
x = np.arange(-range_val, range_val + step, step)
y = np.arange(-range_val, range_val + step, step)
X, Y = np.meshgrid(x, y)

# ---- Rastrigin関数定義 ----
Z = A*2 + (X**2 - A*np.cos(2*np.pi*X)) + (Y**2 - A*np.cos(2*np.pi*Y))

# ---- グラフ描画設定 ----
fig = plt.figure(figsize=(12, 5))

# --- 左: 3Dサーフェス図 ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='turbo', edgecolor='none')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Rastrigin Function (3D Surface)')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
ax1.view_init(30, 45)  # 視点設定

# --- 右: 等高線図 ---
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=50, cmap='turbo')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Rastrigin Function (Contour)')
ax2.axis('equal')
fig.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()