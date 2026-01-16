from typing import Optional

import wandb
try:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws
except ImportError as e:
    raise ImportError(f"{e}\nRUN: pip install wandb-workspaces")


def build_custom_wandb_workerspace(project: str, entity: Optional[str] = None):
    entity = wandb.Api().default_entity if entity is None else entity

    workspace = ws.Workspace(
        name="Jie Cheng's verl Workspace Template",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Validation ACC",
                panels=[
                    wr.LinePlot(y=["val/test_score/math500"], range_y=(0.7, None), title='MATH500 ACC'),
                    wr.LinePlot(y=["val/test_score/aime2024"], range_y=(0., None), title='AIME2024 ACC'),
                ],
                is_open=True,
                panel_settings=ws.SectionPanelSettings(
                    smoothing_type="none",
                ),
                layout_settings=ws.SectionLayoutSettings(
                    columns=3, rows=1,
                ),
            ),
            ws.Section(
                name="Reward & Return",
                panels=[
                    wr.LinePlot(
                        y=["instant_rewards/verifiable_reward/mean"], 
                        title='Verifiable Reward',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["instant_rewards/outcome_reward/mean"], 
                        title='Outcome Reward',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["critic/kl", "instant_rewards/kl/mean"], 
                        title='KL Penalty',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["critic/rewards/mean"], 
                        title='Sequence Reward',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["critic/returns/mean"], 
                        title='Return',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                ],
                is_open=True,
                panel_settings=ws.SectionPanelSettings(
                    smoothing_type="exponential",
                    smoothing_weight=0.8,
                ),
                layout_settings=ws.SectionLayoutSettings(
                    columns=3, rows=2,
                ),
            ),
            ws.Section(
                name="High-Level Pattern-Related",
                panels=[
                    wr.LinePlot(
                        y=["high_level_scores/repeat_score/mean"], 
                        title='Repeatness',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["high_level_scores/reflection_pattern_score/mean"], 
                        title='Reflection Pattern',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=[
                            "response_length/incorrect/mean",
                            "response_length/whole/mean",
                            "response_length/correct/mean",
                        ], 
                        title='Response Length',
                        legend_template='[[${x}: ${y}]] ${run:displayName}, ${metricName}',
                    ),
                ],
                is_open=True,
                panel_settings=ws.SectionPanelSettings(
                    smoothing_type="exponential",
                    smoothing_weight=0.8,
                ),
                layout_settings=ws.SectionLayoutSettings(
                    columns=3, rows=1,
                ),
            ),
            ws.Section(
                name="Group ACC",
                panels=[
                    # model capabilities
                    wr.LinePlot(
                        y=["group_info/pass_at_n/whole_batch/mean", "group_info/pass_at_n/retained_batch/mean"],
                        title='Pass@n',
                        legend_template='[[${x}: ${y}]] ${run:displayName}, ${metricName}',
                    ),
                    wr.LinePlot(
                        y=["group_info/avg_n/whole_batch/mean", "group_info/avg_n/retained_batch/mean"],
                        title='Avg@n',
                        legend_template='[[${x}: ${y}]] ${run:displayName}, ${metricName}',
                    ),
                ],
                is_open=True,
                panel_settings=ws.SectionPanelSettings(
                    smoothing_type="exponential",
                    smoothing_weight=0.8,
                ),
                layout_settings=ws.SectionLayoutSettings(
                    columns=3, rows=1,
                ),
            ),
            ws.Section(
                name="Stability",
                panels=[
                    # focus on repeatness / uncommon response length rise
                    wr.LinePlot(
                        y=["response_length/incorrect/clip_ratio", "response_length/correct/clip_ratio"], 
                        title='Response Clip Ratio',
                        legend_template='[[${x}: ${y}]] ${run:displayName}, ${metricName}',
                    ),
                    wr.LinePlot(
                        y=["high_level_scores/high_repeatness_ratio/incorrect", "high_level_scores/high_repeatness_ratio/correct"], 
                        title='High Repeatness Ratio',
                        legend_template='[[${x}: ${y}]] ${run:displayName}, ${metricName}',
                    ),
                ],
                is_open=True,
                panel_settings=ws.SectionPanelSettings(
                    smoothing_type="exponential",
                    smoothing_weight=0.8,
                ),
                layout_settings=ws.SectionLayoutSettings(
                    columns=3, rows=1,
                ),
            ),
            ws.Section(
                name="Misc",
                panels=[
                    # debug
                    wr.LinePlot(
                        y=["timing_s/step", "timing_s/rollout"], 
                        title='timing',
                        legend_template='[[${x}: ${y}]] ${run:displayName}, ${metricName}',
                    ),
                    wr.LinePlot(
                        y=["instant_rewards/orm_match_vr"], 
                        title='ORM Match VR?',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["num_of_steps/mean"], 
                        title='Num of Steps',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["curriculum_learning/retain_ratio"], 
                        title='Retained Ratio of Curriculum Learning',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["_runtime"], 
                        title='Runtime (hours)', 
                        custom_expressions=['_runtime/3600'], 
                        smoothing_type='none',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                ],
                is_open=True,
                panel_settings=ws.SectionPanelSettings(
                    smoothing_type="exponential",
                    smoothing_weight=0.8,
                ),
                layout_settings=ws.SectionLayoutSettings(
                    columns=3, rows=2,
                ),
            ),
            ws.Section(
                name="Loss",
                panels=[
                     # stable
                    wr.LinePlot(
                        y=["actor/pg_loss"], 
                        title='Policy Gradient Loss',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["actor/entropy_loss"], 
                        title='Entropy Loss',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                    wr.LinePlot(
                        y=["actor/avg_ratio"], 
                        title='Avg IS Ratio',
                        legend_template='[[${x}: ${y}]] ${run:displayName}',
                    ),
                ],
                is_open=False,
                panel_settings=ws.SectionPanelSettings(
                    smoothing_type="exponential",
                    smoothing_weight=0.8,
                ),
                layout_settings=ws.SectionLayoutSettings(
                    columns=3, rows=1,
                ),
            ),
        ],
        settings=ws.WorkspaceSettings(
            x_min=0,
            ignore_outliers=False,
            max_runs=20,
            tooltip_number_of_runs="all_runs",
        ),
        runset_settings=ws.RunsetSettings(
            filters=[
                (ws.Metric('CreatedTimestamp') >= '2025-03-28T16:00:00.000Z'),
            ],
        ),
    ).save()

    # print('Custom workerspace is at: ', workspace.url)
    return workspace.url


if __name__ == '__main__':
    import argparse
    from typing import Optional

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='pure_verl')
    parser.add_argument('--entity', type=str, default=None)
    args = parser.parse_args()

    build_custom_wandb_workerspace(project=args.project, entity=args.entity)