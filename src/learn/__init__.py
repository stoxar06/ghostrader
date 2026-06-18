"""Learn-from-video pipeline — transcribe a strategy video, then HONESTLY test it.

A clear transcript of a trading strategy is still just a set of TA rules, and this repo
has already shown those have no per-trade edge. So this package does not teach the bot to
"win" from a video; it transcribes the claimed strategy, extracts the rules, and runs them
through the same out-of-sample + multiple-testing harnesses as everything else — turning a
"this setup is 90% accurate!" video into an honest verdict. See `pipeline.py`.
"""
