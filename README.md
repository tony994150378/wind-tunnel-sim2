# 🌊 风洞模拟器 | Wind Tunnel CFD Simulator

基于 **Lattice Boltzmann Method (D3Q19)** 的浏览器端风洞流体动力学模拟器。

![Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge)

## ✨ 功能

- 🧮 **LBM D3Q19 流体求解器** — 专业级格子玻尔兹曼方法
- 🎨 **实时流场可视化** — 粒子追踪 + 截面云图
- 📁 **3D 模型上传** — 支持 OBJ / STL 格式
- 🎯 **内置模型** — 球体、汽车、NACA 翼型
- 🌈 **多种色彩映射** — 彩虹、冷暖、等值线
- 📊 **实时数据** — 雷诺数、步数、FPS 监控
- 🖥️ **纯前端** — 无需后端，浏览器直接运行

## 🚀 GitHub Pages 部署

### 方法一：手动部署

1. Fork 或创建新仓库
2. 上传所有文件到仓库根目录
3. 进入仓库 **Settings → Pages**
4. Source 选择 **Deploy from a branch**
5. Branch 选择 **main**，文件夹选择 **/ (root)**
6. 点击 **Save**，等待几分钟即可访问

### 方法二：命令行部署

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/wind-tunnel-sim.git
cd wind-tunnel-sim

# 复制文件到仓库
cp -r /path/to/wind-tunnel-sim/* .

# 提交并推送
git add .
git commit -m "Deploy wind tunnel simulator"
git push origin main
```

### 方法三：GitHub Actions 自动部署

在仓库中创建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [ main ]
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: '.'
      - id: deployment
        uses: actions/deploy-pages@v4
```

## 📁 文件结构

```
wind-tunnel-sim/
├── index.html      # 主页面
├── style.css       # 样式表
├── app.js          # 核心应用代码
├── .nojekyll       # GitHub Pages 禁用 Jekyll
└── README.md       # 说明文档
```

## 🎮 使用方法

1. 打开网页，选择内置模型或上传自己的 OBJ/STL 文件
2. 调整模拟参数（风速、粘度、网格尺寸）
3. 点击 **▶ 开始模拟**
4. 观察流场可视化效果
5. 鼠标拖拽旋转视角，滚轮缩放

## 🔬 技术原理

### Lattice Boltzmann Method (LBM)

LBM 是一种基于介观尺度的流体模拟方法，通过模拟粒子分布函数在格子上的演化来求解 Navier-Stokes 方程。

- **格子模型**: D3Q19（3维空间，19个离散速度方向）
- **碰撞算子**: BGK（单松弛时间近似）
- **边界条件**: Zou-He 速度入口、零梯度出口、反弹格式固壁

### 可视化

- **粒子追踪**: GPU 加速的点精灵粒子系统，颜色映射速度大小
- **截面云图**: 3D 纹理采样的截面速度场渲染
- **色彩映射**: 彩虹 / 冷暖 / 等值线三种方案

## 🌐 浏览器兼容性

| 浏览器 | 最低版本 |
|--------|----------|
| Chrome | 89+ |
| Firefox | 108+ |
| Safari | 16.4+ |
| Edge | 89+ |

需要支持 **WebGL2** 和 **ES Modules (Import Maps)**。

## 📝 License

MIT License
