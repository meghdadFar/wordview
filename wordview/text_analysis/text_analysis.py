from typing import Set, Dict
from wordview.text_analysis.core_text_analysis import do_analysis, plotly_wordcloud
import pandas
import plotly.graph_objs as go
import plotly.figure_factory as ff


class TextStatsPlots:
    def __init__(self,
        df: pandas.DataFrame,
        distributions: Set=['doc_len', 'word_frequency_zipf'],
        pos_tags: Set=['NN', 'VB', 'JJ']
        ) -> None:
        self.df = df
        self.analysis_res = do_analysis(df=self.df,
                               out_dir='output_dir',
                               text_col='text',
                               label_cols=[('label', 'categorical')])
        self.distributions = distributions
        self.dist_plots = self.create_dist_plots()
        self.pos_tags = pos_tags
        self.pos_plots = 

    def create_dist_plots(self):
        """Returns distribution plots for items in `self.distributions`."""
        dist_setup = {
                    'paper_bgcolor': '#007A78',
                    }
        res = {}
        if 'doc_len' in self.distributions:
            fig_doc_len_dist = ff.create_distplot([self.analysis_res.doc_lengths], group_labels=["distplot"], colors=["blue"])
            fig_doc_len_dist.update_layout(dist_setup)
            res['doc_len'] = fig_doc_len_dist 

        if 'word_frequency_zipf' in self.distributions:
            fig_w_freq = go.Figure()
            fig_w_freq.add_trace(go.Scattergl(x=self.analysis_res.zipf_x, y=self.analysis_res.zipf_y_emp, mode='markers'))
            fig_w_freq.add_trace(go.Scattergl(x=self.analysis_res.zipf_x, y=self.analysis_res.zipf_y_theory, mode='markers'))
            fig_w_freq.update_layout(dist_setup)
            res['word_frequency_zipf'] = fig_w_freq

        return res

    def pos_plots(self):
        word_cloud_setup = {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                            'xaxis_showgrid':False,
                            'yaxis_showgrid':False,
                            'xaxis_zeroline':False,
                            'yaxis_zeroline':False,
                            'yaxis_visible':False,
                            'yaxis_showticklabels':False,
                            'xaxis_visible':False,
                            'xaxis_showticklabels':False,
                            }
        res = {}
        if 'NN' in self.pos_tags:
            res['noun_cloud'] = go.Figure(plotly_wordcloud(token_count_dic=self.analysis_res.nns))
            res['verb_cloud'] = go.Figure(plotly_wordcloud(token_count_dic=self.analysis_res.vs))
            res['adj_cloud'] = go.Figure(plotly_wordcloud(token_count_dic=self.analysis_res.jjs))

        for _, fig in res:
            fig.update_layout(word_cloud_setup)
        
        return res