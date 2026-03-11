# EDA Subprocess Tool 路线图

## 1. 目标与边界
- 用 **Python标准库** 实现一个可嵌入Agent的EDA执行工具。
- 优先支持本地进程；同时通过后端抽象支持将来接入LSF/Slurm等调度系统。
- 对复杂EDA执行链路提供：环境准备、前置检查（含license check）、超时、重试、取消、清理、日志尾部收集。

## 2. 核心设计
1. **任务模型**
   - `TaskRequest`: command/cwd/env/login_shell/shell_init_script/timeout/retry/preflight。
   - `TaskState`: pending/running/succeeded/failed/timeout/cancelled。
   - `TaskResult`: 输出尾部、返回码、错误信息、耗时等。
2. **执行后端抽象**
   - `ExecutionBackend` 协议：`submit/poll/read_incremental_output/cancel/cleanup`。
   - 默认 `LocalSubprocessBackend` 使用 `subprocess.Popen` + 线程泵读stdout/stderr。
3. **Agent工具入口**
   - `EdaSubprocessTool.submit()` 异步提交。
   - `EdaSubprocessTool.run_sync()` 同步阻塞执行。
   - `status/get_result/cancel` 用于轮询、收敛和中止。

## 3. 关键问题与实现策略
### 3.1 Login Shell环境差异
- 问题：`subprocess` 默认不加载 `.bashrc/.profile`，EDA工具常依赖module/source脚本。
- 方案：
  - `login_shell=True` 时用 `bash -lc` 运行命令。
  - `shell_init_script` 支持先 `source`，通过 `env -0` 捕获完整环境注入子进程。

### 3.2 任务监控与完成判定
- 周期轮询 `backend.poll()` 获取完成状态与返回码。
- 增量拉取 stdout/stderr，保留可配置长度尾部避免内存膨胀。
- 判定规则：
  - rc=0 => `succeeded`
  - rc!=0 => `failed`
  - 达到timeout => `timeout`
  - 用户取消 => `cancelled`

### 3.3 重试与恢复
- 使用 `RetryPolicy(max_retries, backoff_seconds)`。
- 每次attempt都重跑preflight，避免坏环境下盲目重试。
- 仅在未成功/未取消时按策略退避重试。

### 3.4 资源清理
- 本地后端使用 `start_new_session=True`，便于按进程组终止。
- 取消时先 `SIGTERM`，超时收敛再 `SIGKILL`。
- 无论成功失败都进入 `finally` 调用 `cleanup()`。

### 3.5 前置检查（License / 环境）
- `preflight_checks` 为命令数组列表。
- 典型用法：
  - `['which', 'calibre']`
  - `['bash', '-lc', 'lmutil lmstat -a | grep FEATURE_X']`
- 任一检查失败则任务不启动，状态标记失败并返回错误信息。

## 4. 后续扩展建议
- 增加 `SchedulerBackend`（封装`sbatch/squeue/scancel`或`bsub/bjobs/bkill`）。
- 增加结构化日志落盘（每任务独立目录）。
- 增加并发配额（CPU/GPU/license token aware semaphore）。
- 增加结果工件（波形/版图/报告）归档与摘要接口。
