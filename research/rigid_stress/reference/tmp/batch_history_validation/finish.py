#!/usr/bin/env python3
"""Summarize CPU ablations and pack one reproducible deliverable."""
import csv, json, statistics, zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    data=json.loads((HERE/'batch_history_results.json').read_text())
    groups={}
    for row in data['pipelines']:
        key=(row['case'],row['B'],row['method'])
        groups.setdefault(key,[]).append(row)
    summaries=[]
    for (case,B,method),rows in groups.items():
        mean_seconds=statistics.mean(r['total_seconds'] for r in rows)
        n=B*rows[0]['steps']
        summaries.append({'case':case,'B':B,'method':method,'repetitions':len(rows),
            'steps_per_environment':rows[0]['steps'],
            'mean_env_ms':mean_seconds/n*1000,'env_recoveries_per_second':n/mean_seconds,
            'throughput_min':min(r['env_recoveries_per_second'] for r in rows),
            'throughput_max':max(r['env_recoveries_per_second'] for r in rows),
            'failed_fraction_nonzero':statistics.mean(r['failed_fraction_nonzero'] for r in rows),
            'max_peak_error_percent':max(r['max_peak_error_percent_1Pa_floor'] for r in rows),
            'max_full_relative_residual':max(r['max_full_relative_residual_nonzero'] for r in rows),
            'actual_solved_RHS_columns':sorted(set(r['actual_solved_RHS_columns'] for r in rows)),
            'factor_API_calls':sorted(set(r['factor_API_calls'] for r in rows)),
            'stage_mean_ms_per_env':{k:statistics.mean(r['stage_mean_ms_per_env'][k] for r in rows) for k in rows[0]['stage_mean_ms_per_env']}})
    report={'metadata':{k:v for k,v in data.items() if k not in ['pipelines','microbenchmarks']},
            'summaries':summaries,'microbenchmarks':data['microbenchmarks']}
    (HERE/'batch_history_summary.json').write_text(json.dumps(report,indent=2))
    labels={'serial_direct':'逐场景完整回代','batch_direct':'批量完整回代',
        'serial_legacy':'原时序算法，逐场景运行','batch_legacy_dense':'批量时序，未压缩失败环境',
        'batch_legacy_compact':'批量时序＋失败环境压缩','batch_cached_compact':'批量时序＋压缩＋历史缓存'}
    lines=['CPU 批量回代、失败环境压缩、历史缓存验证',
        data['CPU'],f"自由度={data['mesh']['DOFs']}；四面体={data['mesh']['tets']}；LU非零项={data['factor_nnz']}",
        '完整流程消融保持FP64、8个历史方向、方程相对残差阈值1e-3；严格对照用1e-6。',
        '计时从已装配RHS开始，包括预测、残差检查、打包/回代/散射、历史更新、全域最大应力扫描。',
        '接触映射、独立参考、输出验证不计时。串行路径累加原query的内部计时；批量路径包括编排。',
        '完整测试B=8每环境512步，覆盖完整抓取周期；B=32每环境连续128步，跨环境覆盖不同抓取阶段；实际步数见summary。',
        '环境是同一移动接触轨迹的不同相位、不同幅值，各自独立维护历史；未假设接触对称或固定热点。',
        '粗网格未达到物理应力收敛。 prescribed friction-admissible snapshots，不是真实Genesis/Coulomb动力学。',
        '微基准的FP32检查比较同一FP32因子的串行/批量或稠密/压缩结果，不能解释为FP32相对FP64精度保证。',
        '时序路径相对完整直接求解有容差取舍；不同优化版时序路径使用相同容差。没有GPU或RL吞吐实测。','']
    for case in ['smooth_friction','abrupt_friction','strict_residual']:
        lines.append(case)
        for s in summaries:
            if s['case']==case:
                lines.append(f"B={s['B']:>2} {labels[s['method']]}: {s['mean_env_ms']:.4f} ms/env, {s['env_recoveries_per_second']:.2f} env/s, repeats={s['repetitions']}, peak error max={s['max_peak_error_percent']:.6g}%, failed={s['failed_fraction_nonzero']:.4%}")
        lines.append('')
    lines.append('局部微基准：')
    for m in data['microbenchmarks']['batch']:
        lines.append(f"{m['precision']} B={m['B']} 批量/串行加速={m['speedup']:.3f}x；批量={m['batch']['median_ms']:.4f}ms")
    for m in data['microbenchmarks']['compaction']:
        lines.append(f"{m['precision']} B={m['B']} failed={m['failed_fraction']:.1%}: 压缩加速={m['speedup']:.3f}x；打包={m['pack_only']['median_ms']:.4f}ms；包括分配与散射")
    for m in data['microbenchmarks']['history']:
        lines.append(f"history{m['capacity']}: 投影加速={m['project_speedup']:.3f}x；更新加速={m['append_legacy_median_ms']/m['append_cached_median_ms']:.3f}x；K范数投影差={m['projection_difference_Knorm_after']:.3g}")
    lines+=['','实现：CachedHistory预分配Q/KQ并按FIFO复用列槽，只更新Gram矩阵的一行/列。',
            'Gram=(KQ)^T(KQ)。投影仍解相同小系统，仍重算当前完整K残差决定接受或回代。',
            '压缩只将失败环境的残差列送入同一固定因子，将修正散射回原环境；通过环境索引检查与全域应力参考验证。']
    (HERE/'batch_history_report.txt').write_text('\n'.join(lines))
    with (HERE/'batch_history_table.csv').open('w',newline='') as stream:
        fields=['case','B','method','repetitions','mean_env_ms','env_recoveries_per_second','failed_fraction_nonzero','max_peak_error_percent','max_full_relative_residual']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in summaries:writer.writerow({k:row[k] for k in fields})
    # Input scripts retain their directory layout so imports and ROOT resolve.
    source_files=['output/arbitrary_contact_cpu/arbitrary_contact_test.py','output/arbitrary_contact_cpu/general_peak.py',
        'output/arbitrary_contact_cpu/indexed_contacts.py','output/smooth_friction_cpu/smooth_loads.py',
        'output/smooth_friction_cpu/temporal_benchmark.py','output/eggshell_gripper_cpu.py',
        'output/eggshell_convergence_cpu.py','output/rigid_stress_reference.py','output/rigid_stress_temporal_cpu_test.py']
    local=['benchmark.py','finish.py','batch_history_results.json','batch_history_summary.json','batch_history_report.txt','batch_history_table.csv']
    destination=ROOT/'output/batch_compaction_history_cpu.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in local:archive.write(HERE/name,'tmp/batch_history_validation/'+name)
        for name in source_files:archive.write(ROOT/name,name)
        archive.writestr('README.txt','Install numpy scipy numba threadpoolctl.\nRun: python tmp/batch_history_validation/benchmark.py --repeats 2\nThen: python tmp/batch_history_validation/finish.py\nUse --quick for smoke validation only. Read report for scope and accuracy limits. No GPU/RL benchmark.\n')
    with zipfile.ZipFile(destination) as archive:assert archive.testzip() is None
    print(json.dumps({'summaries':summaries,'zip':str(destination)},indent=2))


if __name__=='__main__':main()
