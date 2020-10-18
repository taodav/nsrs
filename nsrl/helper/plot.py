import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

import visdom

from plotly import tools
import plotly.graph_objs as go


class Plotter(object):
    def __init__(self, experiment_dir, env_name="default", host=None, port=8097, offline=False):
        if not host:
            host = 'localhost'
        print("connecting to host: " + host + ':' + str(port))

        log_file = os.path.join(experiment_dir, "plot")

        self.env_name = env_name

        self.vis = visdom.Visdom(log_to_filename=log_file,
                                 offline=offline,
                                 server='http://' + host, port=port)
        self.plots = {}

    def plot_dict(self, x, value_dict):
        for k, v in value_dict.items():
            self.plot(k, x[:len(v)], v, k, ymin=0)

    def plot(self, var_name, x, y, title_name="Default Plot",
             ymin=None, ymax=None, xmin=None, xmax=None, markers=False,
             linecolor=None, name="default"):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=x, Y=y, env=self.env_name, name=name,
                opts=dict(
                    title=title_name,
                    xlabel='training_steps',
                    ylabel=var_name,
                    xtickmin=xmin,
                    xtickmax=xmax,
                    ytickmin=ymin,
                    ytickmax=ymax,
                    markers=markers,
                    linecolor=linecolor
            ))
        else:
            self.vis.line(X=x, Y=y,
                          env=self.env_name, win=self.plots[var_name],
                          update='append', name=name,
                          opts=dict(
                                title=title_name,
                                xlabel='training_steps',
                                ylabel=var_name,
                                xtickmin=xmin,
                                xtickmax=xmax,
                                ytickmin=ymin,
                                ytickmax=ymax,
                                markers=markers,
                                linecolor=linecolor
                        ))

    def plot_text(self, var_name, text):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.text(text, env=self.env_name)
        else:
            self.vis.text(text,
                          env=self.env_name, win=self.plots[var_name])

    def plot_mpl_fig(self, var_name, fig, title_name='Default MPL plot', replace=False):
        fig = tools.mpl_to_plotly(fig)
        fig['layout'].update(width=650, height=500, title=title_name, showlegend=False)

        if not replace:
            if var_name not in self.plots or not replace:
                self.plots[var_name] = self.vis.plotlyplot(fig, env=self.env_name)
            else:
                self.vis.plotlyplot(fig, env=self.env_name, win=self.plots[var_name])
        else:
            self.plots[var_name] = self.vis.plotlyplot(fig, env=self.env_name)

        self.vis.update_window_opts(win=self.plots[var_name],
                                    opts=dict(
                                        width=650,
                                        height=500
                                    ))

    def plot_mpl_plt(self, var_name, plt, title_name='Default MPL plot', replace=False):
        if not replace:
            if var_name not in self.plots:
                self.plots[var_name] = self.vis.matplot(plt, env=self.env_name, opts=dict(
                    title_name=title_name
                ))
            else:
                self.vis.matplot(plt, win=self.plots[var_name], env=self.env_name, opts=dict(
                    title_name=title_name
                ))
        else:
            self.vis.matplot(plt, env=self.env_name, opts=dict(
                title_name=title_name
            ))
        plt.close()

    def fig2data(self, fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = np.roll(buf, 3, axis=2)
        return buf

    def plot_quiver(self, var_name, up, down, left, right, title_name="default"):
        x = right - left
        y = up - down
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.quiver(X=x, Y=y, env=self.env_name, opts=dict(
                title=title_name,
                normalize=1,
                layoutopts=dict(
                    plotly=dict(
                        yaxis=dict(autorange='reversed')
                    ))))
        else:
            self.vis.quiver(X=x, Y=y, win=self.plots[var_name], env=self.env_name, opts=dict(
                title=title_name
            ))


    def plot_scatter(self, var_name, x, y=None, title_name="default"):
        self.plots[var_name] = self.vis.scatter(x, y, env=self.env_name, opts=dict(
            title=title_name,
            xtickmin=0,
            xtickmax=16,
            ytickmin=0,
            ytickmax=16,
            ztickmin=0
        ))

    def plot_heatmap(self, var_name, x, title_name="default"):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.heatmap(X=x, env=self.env_name, opts=dict(
                colormap='Viridis',
                title=title_name
            ))
        else:
            self.vis.heatmap(X=x, win=self.plots[var_name], env=self.env_name, opts=dict(
                title=title_name
            ))

    def plot_mapping_heatmap(self, var_name, heatmaps, title_name='default', cols=4):
        nrows = (len(heatmaps) + (cols - 1)) // cols
        fig = tools.make_subplots(rows=nrows, cols=cols, subplot_titles=[title for title, _ in heatmaps])
        for i, (title, x) in enumerate(heatmaps, 1):
            hmap = go.Heatmap(z=x, colorscale='Viridis')
            col = i % cols
            col = col if col != 0 else cols

            fig.append_trace(hmap, (i + (cols - 1)) // cols, col)

        fig['layout'].update(width=300 * cols, height=400 * nrows, title=title_name, showlegend=False)
        for e in fig.layout:
            if 'yaxis' in e:
                fig['layout'][e].update(autorange='reversed')
        self.plots[var_name] = self.vis.plotlyplot(fig, env=self.env_name)
        self.vis.update_window_opts(win=self.plots[var_name],
                                    opts=dict(
                                        width=300 * cols,
                                        height=400 * nrows
                                    ))

    def plot_plotly_fig(self, var_name, fig, title_name='default'):
        layout = go.Layout(title=title_name)
        fig = go.Figure(data=fig['data'], layout=layout)
        self.plots[var_name] = self.vis.plotlyplot(fig, env=self.env_name)

    def plot_image(self, var_name, image, title_name="default"):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)

        if var_name not in self.plots:
            self.plots[var_name] = self.vis.image(image, env=self.env_name, opts=dict(
                title=title_name
            ))
        else:
            self.vis.image(image, env=self.env_name, win=self.plots[var_name])

    def plot_video(self, var_name, video):
        self.plots[var_name] = self.vis.video(video, env=self.env_name)

def scatter_3d(x, y, z, color='blue'):
    trace = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=4, symbol="cross", color=color)
    )
    return trace

def scatter_3d_multi_color(point_list):
    """

    :param point_list: list of dicts {x: v, y: v, z:v, color: "color"}
    :return:
    """
    data = [scatter_3d(d['x'], d['y'], d['z'], color=d['color']) for d in point_list]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

def create_loss_figure(loss_dict, x, n, title='default'):
    nrows = len(loss_dict.items()) // 2
    if len(loss_dict.items()) > nrows:
        nrows += 1
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(8, 12))
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    for (key, value), ax in zip(loss_dict.items(), axes.flatten()):
        graph_data(ax, key + " " + str(n),
                   x,
                   value)

    file_path = os.path.join(os.getcwd(), "plots", title + str(n) + '.png')

    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, bbox_inches='tight')
    plt.clf()

def graph_data(ax, title, x, y):
    ax.plot(x, y, lw=2)
    ax.set_title(title)

def replay_plot(fname, host='localhost', port=8098):
    vis = visdom.Visdom(server='http://' + host, port=port)
    vis.replay_log(fname)
    return vis

def get_visdom_data(data, titles):
    results = {}
    inv_titles = {value: key for key, value in titles.items()}
    for window_name, content in data['jsons'].items():
        if 'title' in content and content['title'] in titles.values():
            results[inv_titles[content['title']]] = content['content']['data']
    return results

def parse_visdom_plot_dir(plot_dir, titles, trials=1):
    """
    Parses a .visdom directory that includes saved plots.
    :param plot_dir: directory path to plot
    """
    results = {}
    for fname in os.listdir(plot_dir)[:trials]:
        if fname.endswith('.json'):
            with open(os.path.join(plot_dir, fname)) as json_file:
                data = json.load(json_file)
                found = get_visdom_data(data, titles)
                results[fname] = found
    return results

def plot_offline(experiments_dir):
    """
    For offline plotting usage.
    :param experiments_dir: directory with experiment folders inside
    :return: results from parse_func.
    """
    assert os.path.isdir(experiments_dir)
    for experiment_dir in os.listdir(experiments_dir):
        plot_file = os.path.join(experiments_dir, experiment_dir, 'plot')
        json_plot_file = os.path.join(os.path.expanduser('~'), '.visdom', experiment_dir + '.json')
        if not os.path.exists(json_plot_file):
            replay_plot(plot_file)
        # shutil.move(json_plot_file, experiments_dir)


def group_results(mf):
    """
    Groups results by their keys.
    :param mf: visdom JSON
    :return: grouped results
    """
    mf_plots = {k: [] for k in list(mf.values())[0].keys()}
    for fname, res in mf.items():
        for k in mf_plots.keys():
            mf_plots[k].append(res[k][0]['y'][:1000])
    return mf_plots


def plot_means_with_std(x, explr_fac, visited_ratios, title, fig, ax1, ax2, legends, keys, color='orange'):
    avg_mf_exploration_factor = np.average(explr_fac, axis=0)

    mf1, = ax1.plot(x, avg_mf_exploration_factor, color=color)

    avg_mf_ratios_visited = np.average(visited_ratios, axis=0)
    mf2, = ax2.plot(x, avg_mf_ratios_visited, color=color)

    #     y_mins_ef = explr_fac.min(axis=0)
    #     y_max_ef = explr_fac.max(axis=0)
    y_mins_ef = avg_mf_exploration_factor - np.std(explr_fac, axis=0)
    y_max_ef = avg_mf_exploration_factor + np.std(explr_fac, axis=0)
    ax1.fill_between(x, y_mins_ef, y_max_ef, color=color, alpha=0.2)

    y_mins_rv = avg_mf_ratios_visited - np.std(visited_ratios, axis=0)
    y_max_rv = avg_mf_ratios_visited + np.std(visited_ratios, axis=0)
    ax2.fill_between(x, y_mins_rv, y_max_rv, color=color, alpha=0.2)

    legends.append(mf1)
    keys.append(title)

    return mf1, legends, keys


def exploration_plots(x, mf, fig, ax1, ax2, legends, keys, title='default', color='orange'):
    mf_plots = group_results(mf)

    exp_fac_plots = mf_plots['exploration']
    vis_rat_plots = mf_plots['ratio_states']

    explr_fac = np.array(exp_fac_plots)
    visited_ratios = np.array(vis_rat_plots)

    return plot_means_with_std(x, explr_fac, visited_ratios, title, fig, ax1, ax2, legends, keys, color)


def plot_baseline(plot_fname, ax1, ax2, legends, keys, plot=True):
    with open(plot_fname, 'r') as f:
        baseline = json.load(f)
        exp_factor_baseline = np.array([l for l in baseline['exploration_factors'] if l])
        avg_exp_factor_baseline = np.average(exp_factor_baseline, axis=0)

        ratio_visited_baseline = np.array([l for l in baseline['ratios_visited'] if l])
        avg_ratio_visited_baseline = np.average(ratio_visited_baseline, axis=0)

    x = np.arange(0, avg_exp_factor_baseline.shape[0])
    if plot:
        ax1.title.set_text('Exploration factor')
        b1, = ax1.plot(x, avg_exp_factor_baseline, color='blue')
        y_min_baseline = avg_exp_factor_baseline - np.std(exp_factor_baseline, axis=0)
        y_max_baseline = avg_exp_factor_baseline + np.std(exp_factor_baseline, axis=0)
        ax1.fill_between(x, y_min_baseline, y_max_baseline, color='blue', alpha=0.2)

        ax2.title.set_text('Ratio of states visited')
        b2, = ax2.plot(x, avg_ratio_visited_baseline, color='blue')
        y_min_baseline = avg_ratio_visited_baseline - np.std(ratio_visited_baseline, axis=0)
        y_max_baseline = avg_ratio_visited_baseline + np.std(ratio_visited_baseline, axis=0)
        ax2.fill_between(x, y_min_baseline, y_max_baseline, color='blue', alpha=0.2)

        legends.append(b1)
        keys.append('Random Baseline')

    return legends, keys, x


if __name__ == "__main__":
    from definitions import ROOT_DIR
    experiment = os.path.join(ROOT_DIR, "experiments", 'ALE', "runs", 'montezumas revenge novelty_reward_with_d_step_q_planning_2020-05-29 12-48-51_8222810/plot')
    replay_plot(experiment)

    # exp_dir_walls = os.path.join(ROOT_DIR, "experiments", 'maze', "runs", 'walls_count_q')
    # plot_offline(exp_dir_walls)
    # plt.rcParams.update({'font.size': 18})
    #
    # plot_dir = os.path.join(ROOT_DIR, "experiments", 'maze', 'results')
    # old_titles = {'exploration': 'Average exploration factor over 2 episodes',
    #               'ratio_states': 'Average ratio of states visited over 2 episodes'}
    # new_titles = {'exploration': 'Average exploration factor over 1 episodes',
    #               'ratio_states': 'Average ratio of states visited over 1 episodes'}
    # # experiments = [
    # #                ('empty_count_q', 'Count w/ Q-argmax', new_titles),
    # #                ('empty_q_argmax', 'Novelty w/ Q-argmax', new_titles),
    # #                ('empty_1_step', 'Novelty w/ Planning (d=1)', new_titles),
    # #                ('empty_5_step', 'Novelty w/ Planning (d=5)', new_titles),
    # #                ]
    # experiments = [
    #     ('walls_count_q', 'Count w/ Q-argmax', new_titles),
    #     ('walls_q_argmax', 'Novelty w/ Q-argmax', new_titles),
    #     ('walls_1_step', 'Novelty w/ Planning (d=1)', new_titles),
    #     ('walls_5_step', 'Novelty w/ Planning (d=5)', new_titles),
    # ]
    # colors = ['orange', 'purple', 'green', 'red', 'brown', 'cyan']
    #
    # results = {}
    # for exp, title, titles in experiments:
    #     results[exp] = parse_visdom_plot_dir(os.path.join(plot_dir, exp), titles, trials=10)
    #
    # # PLOTTING OUR BASELINE
    # size_maze = 21
    # baseline_data_fname=os.path.join(ROOT_DIR, "experiments", 'maze', 'plots', 'baselines', 'random_agent_wallless_%d.json' % size_maze)
    #
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # # fig.suptitle("Open Labyrinth experiments steps=%d, trials=%d, size_maze=%d" % (n_steps, trials, size_maze), fontsize=16)
    # ax1.set_ylim([0, 1.01])
    # ax1.set_xlabel('environment steps')
    # ax1.set_ylabel('# unique states visited /\n# total states visited', wrap=True)
    #
    # ax2.set_ylim([0, 1.01])
    # ax2.set_xlabel('environment steps')
    # ax2.set_ylabel('proportion of all states visited')
    #
    # ax1.grid(True)
    # ax2.grid(True)
    #
    # fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    # legends = []
    # keys = []
    #
    # legends, keys, x = plot_baseline(baseline_data_fname, ax1, ax2, legends, keys)
    # for color, (name, title, titles) in zip(colors, experiments):
    #     _, legends, keys = exploration_plots(x, results[name], fig, ax1, ax2, legends, keys, title=title, color=color)
    # # fig.legend((l for l in legends), (k for k in keys), 'lower right')
    # plt.show()