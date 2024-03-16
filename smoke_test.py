# ruff: noqa: F403
# ruff: noqa: F405
# ruff: noqa: E722

from tabular_models import *
from tabular_models.pipelines import *

store = DataStore('/data/tabular_data')
for dataset_name in [
    'Australian',  # binary
    'cmc',  # multiclass
    'rainfall_bangladesh[rev]',  # regression
]:
    dataset = store.load_dataset(dataset_name, 0)

    metric = {
        TaskType.REGRESSION: 'd2_absolute_error',
        TaskType.BINARY: 'neg_log_loss',
        TaskType.MULTICLASS: 'neg_log_loss',
    }[dataset.task_type]

    done_results: dict[tuple[str, str], PipelineResult] = {}

    for pipeline_name, split in [
        ('ConstantPredictor', 'none'),
        ('SingleFeature_Num_Correlation', 'none'),
        ('SingleFeature_Num_Linear_Standard', 'none'),
        ('SingleFeature_Num_Linear_QuantileUniform', 'none'),
        ('SingleFeature_Num_Linear_QuantileNormal', 'none'),
        ('SingleFeature_Num_Tree2Leaves', 'none'),
        ('SingleFeature_Num_Tree3Leaves', 'none'),
        ('Linear_l2_OHE_standard', 'none'),
        ('Linear_l2_TE_standard', 'none'),
        ('Linear_l1_num_standard', 'none'),
        ('Linear_l2_num_standard', 'none'),
        ('Linear_l2_num_quantileUniform', 'none'),
        ('Linear_l2_num_quantileNormal', 'none'),
        ('Linear_l2_num_auto', 'none'),
        ('Linear_l2_num_featureWisePreproc', 'none'),
        ('Catboost_baseline', 'cv5_seed0_fold0'),
        ('Catboost_init[Linear_l2_OHE_standard]', 'cv5_seed0_fold0'),
        ('Catboost_init[Linear_l2_TE_standard]', 'cv5_seed0_fold0'),
        ('Catboost_init[Linear_l1_num_standard]', 'cv5_seed0_fold0'),
        ('Catboost_init[Linear_l2_num_standard]', 'cv5_seed0_fold0'),
        ('Catboost_init[Linear_l2_num_quantileUniform]', 'cv5_seed0_fold0'),
        ('Catboost_init[Linear_l2_num_quantileNormal]', 'cv5_seed0_fold0'),
        ('Catboost_init[Linear_l2_num_auto]', 'cv5_seed0_fold0'),
        ('Catboost_init[Linear_l2_num_featureWisePreproc]', 'cv5_seed0_fold0'),
    ]:
        new_result = get_pipeline_object(pipeline_name).run(
            DatasetSplitter.from_string(split).split(dataset), done_results
        )
        done_results[(pipeline_name, split)] = new_result
        print(f'{dataset_name} ({dataset.task_type}), {pipeline_name}')
        try:
            scores_df = new_result.scores.scores_df.filter(
                bootstrap_seed=None
            ).drop('bootstrap_seed')
            print('\n'.join(str(scores_df).split('\n')[6:-1]))
        except:
            print('│ no scores')
