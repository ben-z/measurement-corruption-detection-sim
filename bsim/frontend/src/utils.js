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

function* travelAlongPath(path, slist) {
    // path is a list of [x, y] points.
    // slist is a list of s values (progress along the path, in meters).
    // Yields [x, y, yaw] at each point in slist, where yaw is the
    // direction of the path at that point, assuming linear interpolation.

    let pi = 0;
    let pi_t = 0;
    for (let si = 0; si < slist.length; si++) {
        const s = slist[si];
        const prevS = si > 0 ? slist[si - 1] : 0;
        let remainingS = s - prevS;
        while (remainingS > 0) {
            const [x, y] = path[pi];
            const [next_x, next_y] = path[(pi+1) % path.length];
            const segmentLen = Math.sqrt((next_x - x)**2 + (next_y - y)**2);
            const remainingSegmentLen = segmentLen * (1 - pi_t);
            
            const t = remainingS / segmentLen + pi_t;
            if (t >= 1) {
                remainingS -= remainingSegmentLen;
                pi = (pi + 1) % path.length;
                pi_t = 0;
            } else {
                pi_t = t;
                remainingS = 0;
            }
        }
        const [x, y] = path[pi];
        const [next_x, next_y] = path[(pi+1) % path.length];
        const yaw = Math.atan2(next_y - y, next_x - x);
        yield [x + (next_x - x) * pi_t, y + (next_y - y) * pi_t, yaw];
    }
}
module.exports.travelAlongPath = travelAlongPath;

module.exports.frenet2global_path = function frenet2global_path(path, slist, dlist) {
    // path is a list of [x, y] points
    // s is a list of s values (m)
    // d is a list of d values (m)
    // returns a list of [x, y] points

    const globalPath = [];
    const dIterator = dlist[Symbol.iterator]();
    for (const [x, y, yaw] of travelAlongPath(path, slist)) {
        const d = dIterator.next().value;
        globalPath.push(frenet2global_point([0, d], [x, y], yaw));
    }
    return globalPath;
}

function frenet2global_point(frenet, origin, heading) {
    // frenet is [s, d] (in meters)
    // origin is [x, y]
    // heading is in radians
    // returns [x, y]

    const s = frenet[0];
    const d = frenet[1];

    const x = origin[0] + s * Math.cos(heading) - d * Math.sin(heading);
    const y = origin[1] + s * Math.sin(heading) + d * Math.cos(heading);
    return [x, y];
}
module.exports.frenet2global_point = frenet2global_point;

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

class Semaphore {
    /* 
    A semaphore is a counter that can be incremented and decremented.
    If the counter is zero, decrement() returns a Promise that resolves
    when the counter is incremented. If the counter is not zero, 
    decrement() returns a Promise that resolves immediately.
    If the counter is at the maximum, increment() returns a Promise that
    resolves when the counter is decremented. If the counter is not at
    the maximum, increment() returns a Promise that resolves immediately.
    */
    constructor({max = Infinity, initial = 0}) {
        this._count = initial;
        this._resolve = [];
        this._max = max;
        this._uncommitted_count = 0;
    }

    decrement(atomic_op = () => {}) {
        if (this._count > 0) {
            atomic_op();
            if (this._resolve.length) {
                assert(this._count == this._max, "Semaphore count is not at max but there are waiting promises");
                setTimeout(this._resolve.shift(), 0);
                this._uncommitted_count++;
            } else {
                this._count--;
            }
            return Promise.resolve();
        } else {
            return new Promise((resolve) => {
                this._resolve.push(() => { this._uncommitted_count++; atomic_op(); resolve(); });
            });
        }
    }

    increment(atomic_op = () => {}) {
        if (this._count < this._max) {
            atomic_op();
            if (this._resolve.length) {
                assert(this._count == 0, "Semaphore count is not zero but there are waiting promises")
                setTimeout(this._resolve.shift(), 0);
                this._uncommitted_count--;
            } else {
                this._count++;
            }
            return Promise.resolve();
        } else {
            return new Promise((resolve) => {
                this._resolve.push(() => { this._uncommitted_count--; atomic_op(); resolve(); });
            });
        }
    }

    get count() {
        return this._count - this._uncommitted_count;
    }
}
module.exports.Semaphore = Semaphore;

class PromiseBuffer {
    /* 
    A PromiseBuffer is a queue that can be pushed to and popped from.
    If the buffer is empty, pop() returns a Promise that resolves when
    the buffer is pushed to. If the buffer is not empty, pop() returns
    a Promise that resolves immediately.
    If the buffer is full, push() returns a Promise that resolves when
    the buffer is popped from. If the buffer is not full, push() returns
    a Promise that resolves immediately.
    */
    constructor(capacity = Infinity) {
        this._queue = [];
        this._queue_length_semaphore = new Semaphore({max: capacity, initial: 0});
    }

    async push(item) {
        await this._queue_length_semaphore.increment(() => {
        });
            this._queue.push(item);
    }

    async pop() {
        let ret;
        await this._queue_length_semaphore.decrement(() => {
        });
            ret = this._queue.shift();
        return ret;
    }

    get length() {
        assert(this._queue.length == this._queue_length_semaphore.count, `Queue length (${this._queue.length}) does not match semaphore count (${this._queue_length_semaphore.count})`);
        return this._queue.length;
    }

    get capacity() {
        return this._queue_length_semaphore._max;
    }
}
module.exports.PromiseBuffer = PromiseBuffer;
