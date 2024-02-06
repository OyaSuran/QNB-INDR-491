import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def corrplot(data, size_scale=500, marker='s'):
    """

    A kind of heatmap that created from scatter plot.

    Ref: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec

    """

    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )


def heatmap(x, y, **kwargs):
    """

    Heatmap for corrplot.

    """

    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256                                                                  # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color)                                   # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min)           # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1)                                 # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1))                                    # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01           # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1)                                     # bound the position betwen 0 and 1
            return val_position * size_scale

    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)                                 # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1])                                                      # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
        'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )

    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')
    
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1])                               # Use the rightmost column of the plot

        col_x = [0]*len(palette)                                        # Fixed x coordinate for the bars
        bar_y = np.linspace(color_min, color_max, n_colors)             # y coordinates for each of the n_colors bars
        bar_height = bar_y[1] - bar_y[0]

        ax.barh(
            y=bar_y,
            width=[5]*len(palette),                                     # Make bars 5 units wide
            left=col_x,                                                 # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )

        ax.set_xlim(1, 2)                                               # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False)                                                  # Hide grid
        ax.set_facecolor('white')                                       # Make background white
        ax.set_xticks([])                                               # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))           # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right()                                           # Show vertical ticks on the right 


def visualize_nan(data: pd.DataFrame, name: str) -> None:
    """
    
    Visualize nan values for each column with bar plot.
    
    :param data: pandas dataframe.
    :param name: name of dataframe.
    
    """
    
    # set size of matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # we change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    plt.xticks(rotation=45)
    
    # get nan values for each column
    df_nan = pd.DataFrame(isna().mean(axis=0), columns=["Count"])
    
    # get variables as column
    df_nan = df_nan.reset_index()
    
    # plot missing values only count > 0
    sns.barplot(x="index", y="Count", data=df_nan[df_nan["Count"] > 0])
    
    # plot details
    ax.set_title(f"{name} data nan values by columns", fontsize=12)
    ax.set_xlabel('Columns',   fontsize=12)
    ax.set_ylabel('Nan Count', fontsize=12)
    
    # print info
    print(f"{len(df_nan['Count'].unique()) - 1}/{df_nan.shape[0] - 1} columns have missing values.")
    
    # display
    plt.show()
        
        
def plot_heatmap_v1(data: pd.DataFrame) -> None:
    """

    Plot heatmap with correlations in data.

    :param data: input data

    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(48, 48))

    sns.heatmap(data.corr(), annot=True, cbar=False, annot_kws={"fontsize" : 20})

    plt.title("Correlation Heatmap V1")
    plt.show()


def plot_heatmap_v2(data: pd.DataFrame) -> None:
    """

    Plot heatmap with correlations in data.

    :param data: input data

    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(32, 32))

    corrplot(df_train_var.corr(), size_scale=1024, marker='s')

    plt.title("Correlation Heatmap V2")
    plt.show()


def plot_kde(data: pd.DataFrame, columns: list):
    """
    
    Plot Kernel density estimation plots for given columns and dataframe.

    :param data     : input data
    :param columns  : columns to plot kde in same plot
    
    """
    plt.figure(figsize=(16, 8))
    sns.set_style("white")
    
    # create random color palette for kde
    random_palette = ['#%06X' % random.randint(0,256**3-1) for i in range(len(columns))]

    # set random color palette to seaborn
    sns.set_palette(palette=random_palette)
    
    for index, column in enumerate(columns):
        sns.kdeplot(data=data, x=column, shade=True , color=f"C{index}")
    
    # remove right and top edges of plot
    sns.despine(top=True, right=True, left=False, bottom=False)
    
    plt.legend(columns)
    plt.xlabel(" , ".join(columns), fontsize = 13)
    plt.ylabel("Density", fontsize = 13)
    
    plt.show()


def plot_scatter(data: pd.DataFrame, x: str, y: str, hue: str) -> None:
    """
    
    Plot scatter plot between x and y features.

    :param data : input data
    :param x    : column on x axis
    :param y    : column on y axis
    :param hue  : categorical column 
    
    """
    
    plt.figure(figsize=(16, 8))
    sns.set_style("white")

    sns.scatterplot(data=data, x=x, y=y, hue=hue)

    # remove right and top edges of plot
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


def plot_count_plot(data: pd.DataFrame, x: str, hue: str) -> None:
    """
    
    Plot values count for categorical column x.

    :param data : input data
    :param x    : column to plot values count
    :param hue  : categorical column 
    
    """
    
    plt.figure(figsize=(16, 8))

    sns.color_palette('colorblind')
    sns.countplot(data=data, x=x, hue=hue)

    sns.set_style("white")
    sns.despine(top=True, right=True, left=False, bottom=False)

    plt.title(x)
    plt.show()


def plot_pie_plot(data: pd.DataFrame, col: str) -> None:
    """
    
    Plot pie plot for categorical column x.

    :param data : input data
    :param col  : categorical column
    
    """
    
    x = list(data[col].value_counts().index)
    y = list(data[col].value_counts())
    
    plt.figure(figsize=(16, 16))

    # set color palette for seaborn
    colors = sns.color_palette('colorblind')

    # create pie plot with matplotlib
    patches, texts = plt.pie(y, colors=colors)

    plt.legend(patches, x, loc="best")
    plt.title(col)
    plt.show()


def plot_nan_count(data: pd.DataFrame, name: str) -> None:
    """
    
    Visualize nan values for each column with bar plot.
    
    :param data: pandas dataframe.
    :param name: name of dataframe.
    
    """
    
    # set size of matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # we change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    plt.xticks(rotation=45)
    
    # get nan values for each column
    df_nan = pd.DataFrame(isna().mean(axis=0), columns=["Count"])
    
    # get variables as column
    df_nan = df_nan.reset_index()
    
    # plot missing values only count > 0
    sns.barplot(x="index", y="Count", data=df_nan[df_nan["Count"] > 0])
    
    # plot details
    ax.set_title(f"{name} data nan values by columns", fontsize=12)
    ax.set_xlabel('Columns',   fontsize=12)
    ax.set_ylabel('Nan Count', fontsize=12)
    
    # print info
    print(f"{len(df_nan['Count'].unique()) - 1}/{df_nan.shape[0] - 1} columns have missing values.")
    
    # display
    plt.show()