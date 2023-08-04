import torch
import numpy as np
import torch.autograd.profiler as profiler
import os


def inference_time(model, loader, config):
    if loader is None:
        inputs = torch.randn((1, 3, 224, 224)).to(config["train"]["device"])
    else:
        if config["data"]["representation"] in [
                'spectrogram', 'melspectrogram'
        ]:
            inputs, attributes = next(iter(loader))
            inputs = inputs.float().to(config["train"]["device"])
            attributes = attributes.float().to(config["train"]["device"])
        else:
            inputs = None
            attributes = None
    model.to(config["train"]["device"])
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(inputs, attributes)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(inputs, attributes)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn


def throughput(model, optimal_batch_size, loader, config):
    if loader is None:
        inputs = torch.randn((1, 3, 224, 224)).to(config["train"]["device"])
    else:
        if config["data"]["representation"] in [
                'spectrogram', 'melspectrogram'
        ]:
            inputs, attributes = next(iter(loader))
            inputs = inputs.float().to(config["train"]["device"])
            attributes = attributes.float().to(config["train"]["device"])
        else:
            inputs = None
            attributes = None
    inputs = inputs.to(config["train"]["device"])
    model.to(config["train"]["device"])
    repetitions = 100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(
                enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(inputs, attributes)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    thrghput = (repetitions * optimal_batch_size) / total_time
    return thrghput


def stats_model(model, loader, config):
    if loader is None:
        inputs = torch.randn((1, 3, 224, 224)).to(config["train"]["device"])
    else:
        if config["data"]["representation"] in [
                'spectrogram', 'melspectrogram'
        ]:
            inputs, attributes = next(iter(loader))
            inputs = inputs.float().to(config["train"]["device"])
            attributes = attributes.float().to(config["train"]["device"])
        else:
            inputs = None
            attributes = None
    inputs = inputs.to(config["train"]["device"])
    model.to(config["train"]["device"])
    # Model Inference
    with profiler.profile(use_cuda=True,
                          profile_memory=True,
                          record_shapes=True,
                          with_flops=True,
                          with_stack=True) as prof:
        with profiler.record_function("model_inference"):
            model(inputs, attributes)
    keys_vals = prof.key_averages()
    final_stats = {}
    for k in keys_vals:
        if k.key == 'model_inference':
            final_stats['cpu_time'] = k.cpu_time_total
            final_stats['gpu_time'] = k.cuda_time_total
            full_flops = 0.0
            cpu_mem = 0.0
            gpu_mem = 0.0
            for evt in k.cpu_children:
                full_flops += evt.flops
                cpu_mem += evt.cpu_memory_usage
                gpu_mem += evt.cuda_memory_usage
            final_stats['cpu_memory'] = cpu_mem
            final_stats['gpu_memory'] = gpu_mem
            final_stats['flops'] = full_flops
    total_params = sum(p.numel() for p in model.parameters())
    final_stats['parameters'] = total_params
    m_inf, s_inf = inference_time(model, loader, config)
    final_stats['inference_time'] = m_inf
    final_stats['inference_time_std'] = s_inf
    final_stats['throughput'] = throughput(model, 16, loader, config)
    # Compute model disk size
    torch.save(model, config["data"]["stats_path"] + 'model.test')
    final_stats['disk_size'] = os.stat(config["data"]["stats_path"] +
                                       'model.test').st_size
    torch.save(final_stats, config["data"]["stats_path"] + "model.stats")
    return final_stats
