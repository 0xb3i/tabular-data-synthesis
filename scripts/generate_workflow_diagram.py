"""
生成 Diffusion-based Tabular Data Synthesis 工作流图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_workflow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # 颜色定义
    colors = {
        'data': '#E3F2FD',           # 浅蓝 - 数据
        'tabular_prep': '#E8F5E9',    # 浅绿 - 表格预处理
        'standard_prep': '#FFF3E0',   # 浅橙 - 标准预处理
        'model': '#F3E5F5',           # 浅紫 - 模型
        'sample': '#FFEBEE',          # 浅红 - 采样
        'eval': '#E0F7FA',            # 浅青 - 评估
        'border_data': '#1976D2',
        'border_tabular': '#388E3C',
        'border_standard': '#F57C00',
        'border_model': '#7B1FA2',
        'border_sample': '#D32F2F',
        'border_eval': '#00838F',
        'arrow': '#455A64',
        'title': '#1565C0',
        'phase_label': '#37474F',
    }

    def draw_box(x, y, width, height, text, color, border_color, fontsize=10, bold=False):
        """绘制圆角矩形框"""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            facecolor=color,
            edgecolor=border_color,
            linewidth=2
        )
        ax.add_patch(box)

        weight = 'bold' if bold else 'normal'
        # 处理多行文本
        lines = text.split('\n')
        line_height = height / (len(lines) + 1)
        for i, line in enumerate(lines):
            ax.text(
                x + width/2, y + height - (i + 1) * line_height,
                line,
                ha='center', va='center',
                fontsize=fontsize, fontweight=weight,
                color='#212121'
            )
        return box

    def draw_arrow(start, end, color='#455A64'):
        """绘制箭头"""
        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=2, mutation_scale=15
            )
        )

    def draw_curved_arrow(start, end, color='#455A64', connectionstyle='arc3,rad=0.3'):
        """绘制弯曲箭头"""
        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=2, mutation_scale=15,
                connectionstyle=connectionstyle
            )
        )

    # 标题
    ax.text(8, 19.5, 'Diffusion-based Tabular Data Synthesis 工作流',
            ha='center', va='center', fontsize=18, fontweight='bold', color=colors['title'])

    # ========== 阶段 1: 数据加载 ==========
    draw_box(5.5, 17.5, 5, 1.2,
             '1. 数据加载\n加载原始数据: X_cat, X_num, y',
             colors['data'], colors['border_data'], fontsize=11, bold=True)

    # ========== 阶段 2: 表格预处理 ==========
    draw_box(2, 14.5, 12, 2.5,
             '2. 表格预处理 [第一阶段]\n'
             '─────────────────────────────\n'
             '• TabularDataController.fit_transform()\n'
             '• 处理器类型: identity | bgm | ft\n'
             '• 只在训练集上 fit (防止数据泄露)',
             colors['tabular_prep'], colors['border_tabular'], fontsize=10, bold=True)

    # 表格预处理策略子框
    draw_box(2.5, 12.3, 3.5, 1.5,
             'identity\n不做处理',
             '#FAFAFA', '#81C784', fontsize=9)
    draw_box(6.5, 12.3, 3.5, 1.5,
             'bgm\n贝叶斯高斯混合模型\n+ log变换',
             '#FAFAFA', '#81C784', fontsize=9)
    draw_box(10.5, 12.3, 3.5, 1.5,
             'ft\n特征标记',
             '#FAFAFA', '#81C784', fontsize=9)

    # ========== 阶段 3: 标准预处理 ==========
    draw_box(2, 9.8, 12, 1.8,
             '3. 标准预处理 [第二阶段]\n'
             'normalization (标准化) + cat_encoding (one-hot/ordinal)',
             colors['standard_prep'], colors['border_standard'], fontsize=11, bold=True)

    # ========== 阶段 4: 模型训练 ==========
    draw_box(2, 6.8, 12, 2.2,
             '4. 模型训练\n'
             '─────────────────────────────\n'
             '• GaussianMultinomialDiffusion 扩散模型\n'
             '• 保存: model.pt, model_ema.pt, loss.csv',
             colors['model'], colors['border_model'], fontsize=11, bold=True)

    # ========== 阶段 5: 采样 ==========
    draw_box(2, 3.5, 12, 2.5,
             '5. 采样\n'
             '─────────────────────────────\n'
             '• 加载训练好的模型，扩散采样\n'
             '• 逆向处理: 标准预处理逆变换 → 表格预处理逆变换\n'
             '• 保存: X_num_train.npy, X_cat_train.npy, y_train.npy',
             colors['sample'], colors['border_sample'], fontsize=11, bold=True)

    # ========== 阶段 6: 评估 ==========
    draw_box(2, 0.5, 5.5, 2.2,
             '6a. ML-efficacy 评估\n'
             '────────────────────\n'
             'CatBoost / MLP\n'
             '合成数据训练→真实测试集评估',
             colors['eval'], colors['border_eval'], fontsize=10, bold=True)

    draw_box(8.5, 0.5, 5.5, 2.2,
             '6b. 相似度评估\n'
             '────────────────────\n'
             'TabSynDex + Table-Evaluator\n'
             '合成数据 vs 真实数据对比',
             colors['eval'], colors['border_eval'], fontsize=10, bold=True)

    # ========== 绘制箭头 ==========
    # 1 → 2
    draw_arrow((8, 17.5), (8, 17))

    # 2 → 3
    draw_arrow((8, 12.3), (8, 11.6))

    # 3 → 4
    draw_arrow((8, 9.8), (8, 9))

    # 4 → 5
    draw_arrow((8, 6.8), (8, 6))

    # 5 → 6a, 6b
    draw_arrow((5.5, 3.5), (4.75, 2.7))
    draw_arrow((10.5, 3.5), (11.25, 2.7))

    # ========== 逆向处理流程标注 ==========
    # 右侧标注：表格预处理逆变换
    ax.annotate(
        '', xy=(14.5, 4.2), xytext=(14.5, 7.9),
        arrowprops=dict(
            arrowstyle='->', color='#D32F2F',
            lw=2, mutation_scale=15,
            linestyle='--'
        )
    )
    ax.text(14.8, 6, '逆向变换\n(采样后)', ha='left', va='center',
            fontsize=9, color='#D32F2F', style='italic')

    # ========== 图例 ==========
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor=colors['border_data'], label='数据加载'),
        mpatches.Patch(facecolor=colors['tabular_prep'], edgecolor=colors['border_tabular'], label='表格预处理 (第1阶段)'),
        mpatches.Patch(facecolor=colors['standard_prep'], edgecolor=colors['border_standard'], label='标准预处理 (第2阶段)'),
        mpatches.Patch(facecolor=colors['model'], edgecolor=colors['border_model'], label='模型训练'),
        mpatches.Patch(facecolor=colors['sample'], edgecolor=colors['border_sample'], label='采样生成'),
        mpatches.Patch(facecolor=colors['eval'], edgecolor=colors['border_eval'], label='评估'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='#BDBDBD')

    plt.tight_layout()
    return fig

def create_detailed_data_flow():
    """创建详细的数据流转图"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 颜色
    c = {
        'input': '#BBDEFB',
        'process': '#C8E6C9',
        'output': '#FFCCBC',
        'border': '#455A64',
        'arrow': '#37474F',
        'label': '#1565C0',
    }

    def draw_node(x, y, w, h, text, color, fontsize=9):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color,
            edgecolor=c['border'],
            linewidth=1.5
        )
        ax.add_patch(box)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h - (i + 0.5) * h/len(lines),
                   line, ha='center', va='center', fontsize=fontsize)

    # 标题
    ax.text(9, 11.5, '表格预处理在数据流中的位置 (详细视图)',
            ha='center', fontsize=16, fontweight='bold', color=c['label'])

    # ========== 训练阶段 ==========
    ax.text(1.5, 10.5, '训练阶段', fontsize=12, fontweight='bold', color='#388E3C')

    # 原始数据
    draw_node(0.5, 9, 2.5, 1.2, '原始数据\nX_cat, X_num, y', c['input'])

    # 表格预处理 (fit_transform)
    draw_node(4, 9, 3, 1.2, 'TabularProcessor\nfit_transform()', c['process'])

    # 预处理后数据
    draw_node(8, 9, 2.5, 1.2, '预处理后数据\n保存到新目录', c['output'])

    # 标准预处理
    draw_node(11.5, 9, 2.5, 1.2, 'Transformations\nnormalize + encode', c['process'])

    # 模型输入
    draw_node(15, 9, 2.5, 1.2, '模型输入\nDataset对象', c['input'])

    # 箭头 - 训练
    ax.annotate('', xy=(4, 9.6), xytext=(3, 9.6),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))
    ax.annotate('', xy=(8, 9.6), xytext=(7, 9.6),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))
    ax.annotate('', xy=(11.5, 9.6), xytext=(10.5, 9.6),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))
    ax.annotate('', xy=(15, 9.6), xytext=(14, 9.6),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))

    # 训练模型
    draw_node(15, 7.2, 2.5, 1.2, '扩散模型\n训练', c['process'])
    ax.annotate('', xy=(16.25, 8.2), xytext=(16.25, 9),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))

    # 保存模型标注
    ax.text(16.25, 6.5, 'model.pt', ha='center', fontsize=8, style='italic')

    # ========== 采样阶段 ==========
    ax.text(1.5, 5.5, '采样阶段', fontsize=12, fontweight='bold', color='#D32F2F')

    # 随机噪声
    draw_node(0.5, 4, 2.5, 1.2, '随机噪声\nz ~ N(0,1)', c['input'])

    # 扩散采样
    draw_node(4, 4, 3, 1.2, '扩散采样\ndiffusion.sample()', c['process'])

    # 合成数据 (编码形式)
    draw_node(8, 4, 2.5, 1.2, '合成数据\n(编码形式)', c['output'])

    # 标准预处理逆变换
    draw_node(11.5, 4, 2.5, 1.2, 'inverse_transform\n标准预处理逆变换', c['process'])

    # 表格预处理逆变换
    draw_node(15, 4, 2.5, 1.2, 'inverse_transform\n表格预处理逆变换', c['process'])

    # 箭头 - 采样
    ax.annotate('', xy=(4, 4.6), xytext=(3, 4.6),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))
    ax.annotate('', xy=(8, 4.6), xytext=(7, 4.6),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))
    ax.annotate('', xy=(11.5, 4.6), xytext=(10.5, 4.6),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))
    ax.annotate('', xy=(15, 4.6), xytext=(14, 4.6),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))

    # 模型加载
    ax.annotate('', xy=(5.5, 5.2), xytext=(16.25, 7.2),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5,
                             connectionstyle='arc3,rad=-0.3', linestyle='--'))
    ax.text(11, 6.5, '加载模型', fontsize=8, ha='center', style='italic')

    # 最终输出
    draw_node(15, 2, 2.5, 1.2, '最终合成数据\n(人类可读)', c['output'])
    ax.annotate('', xy=(16.25, 3.2), xytext=(16.25, 4),
               arrowprops=dict(arrowstyle='->', color=c['arrow'], lw=1.5))

    # ========== 关键说明 ==========
    ax.text(1, 0.8, '关键点:', fontsize=11, fontweight='bold')
    ax.text(1, 0.3, '• 表格预处理是第一阶段，在标准预处理之前执行', fontsize=10)
    ax.text(1, -0.2, '• TabularProcessor.fit() 只在训练集上进行，防止数据泄露', fontsize=10)
    ax.text(9, 0.3, '• 采样时需要依次逆向变换：先标准预处理逆变换，再表格预处理逆变换', fontsize=10)
    ax.text(9, -0.2, '• 处理器状态保存在 processor_state/ 目录，可复用', fontsize=10)

    plt.tight_layout()
    return fig

def main():
    # 创建输出目录
    import os
    output_dir = 'outputs/workflow_diagrams'
    os.makedirs(output_dir, exist_ok=True)

    # 生成主工作流图
    fig1 = create_workflow_diagram()
    fig1.savefig(f'{output_dir}/main_workflow.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig1.savefig(f'{output_dir}/main_workflow.pdf', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"主工作流图已保存到: {output_dir}/main_workflow.png")

    # 生成详细数据流图
    fig2 = create_detailed_data_flow()
    fig2.savefig(f'{output_dir}/data_flow_detail.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig2.savefig(f'{output_dir}/data_flow_detail.pdf', bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"详细数据流图已保存到: {output_dir}/data_flow_detail.png")

    plt.close('all')
    print("\n所有图形已生成完成!")

if __name__ == '__main__':
    main()
