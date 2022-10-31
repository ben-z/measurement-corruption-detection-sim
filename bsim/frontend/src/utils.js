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

module.exports.matrixMultiply = function matrixMultiply(m1, m2) {
    // a is a 2D array of numbers
    // b is a 2D array of numbers
    // returns a 2D array of numbers
    // a and b must have the correct dimensions for matrix multiplication
    // Derived from https://stackoverflow.com/a/27205510/4527337

    const result = [];
    for (const i = 0; i < m1.length; i++) {
        result[i] = [];
        for (const j = 0; j < m2[0].length; j++) {
            let sum = 0;
            for (const k = 0; k < m1[0].length; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

module.exports.generateCircleApproximation = function generateCircleApproximation(center, radius, numPoints) {
    // center is [x, y]
    // radius is a number
    // numPoints is an integer
    // returns a list of points on the circle

    const points = [];
    for (let i = 0; i < numPoints; i++) {
        const angle = 2 * Math.PI * i / numPoints;
        points.push([
            center[0] + radius * Math.cos(angle),
            center[1] + radius * Math.sin(angle),
        ]);
    }
    return points;
}

module.exports.approxeq = function approxeq(a, b, epsilon=1e-6) {
    return Math.abs(a - b) < epsilon;
}

module.exports.predSlice = function predSlice(arr, predStart, predEnd) {
    // predSlice is the same arr.slice, except it finds the start and end indices
    // based on predStart and predEnd.
    let start = predStart || 0;
    let end = predEnd || arr.length;

    if (typeof predStart === 'function') {
        start = arr.findIndex(predStart);
        if (start === -1) {
            start = 0;
        }
    }
    if (typeof predEnd === 'function') {
        end = arr.findIndex(predEnd);
        if (end === -1) {
            end = arr.length;
        }
    }

    return arr.slice(start, end);
}
