def profile_function(func):
    def wrapper(*args, **kwargs):
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
        stats.print_stats(20)
        return result
    return wrapper
