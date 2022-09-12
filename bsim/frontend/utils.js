module.exports.mySetInterval = function mySetInterval(callback, delay) {
    // Takes in a callback, which returns a promise, and a delay in ms
    let isCancelled = false;
    function loop() {
        if (isCancelled) {
            return;
        }
        const start = Date.now();
        callback().then(() => {
            if (isCancelled) {
                return;
            }
            setTimeout(loop, Math.max(0, delay - (Date.now() - start)));
        });
    }
    loop();
    return {
        cancel: () => {
            isCancelled = true;
        }
    };
}