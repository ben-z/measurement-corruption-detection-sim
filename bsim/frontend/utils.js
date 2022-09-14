module.exports.mySetInterval = function mySetInterval(callback, delay) {
    // setInterval, but runs at most as fast as the callback takes to run.
    // `callback` should return a promise.
    // `delay` is in milliseconds.

    let isCancelled = false;
    let start = Date.now(); // ms
    function loop() {
        if (isCancelled) {
            return;
        }
        callback().then(() => {
            if (isCancelled) {
                return;
            }
            const now = Date.now();
            const nextStart = start + delay;
            const remainingTime = nextStart - now;
            if (remainingTime < 0) {
                // console.debug(`WARNING: loop is running behind schedule by ${-remainingTime} ms`);
                start = now;
            } else {
                start = nextStart;
            }
            setTimeout(loop, Math.max(0, remainingTime));
        });
    }
    loop();
    return {
        cancel: () => {
            isCancelled = true;
        }
    };
}