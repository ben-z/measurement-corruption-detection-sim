const WebSocketAsPromised = require('websocket-as-promised');

function ensureSucceeds(res) {
    // Ensures that `res` doesn't have an `error` field
    if (res.error) {
        throw new Error(res.error);
    }

    return res;
}

const egos = {
    ego1: {},
}

const exported = module.exports = {}

const WEBSOCKET_OPTIONS = {
    packMessage: data => JSON.stringify(data),
    unpackMessage: data => JSON.parse(data),
    attachRequestId: (data, requestId) => Object.assign({id: requestId}, data), // attach requestId to message as `id` field
    extractRequestId: data => data && data.id,                                  // read requestId from message `id` field
}

async function main() {
    // Initialize world
    const worldSocket = new WebSocketAsPromised('ws://localhost:8765/world', WEBSOCKET_OPTIONS);
    await worldSocket.open();
    console.log("World socket opened: ", worldSocket);
    ensureSucceeds(await worldSocket.sendRequest({command: 'reset'}));
    for (const ego in egos) {
        ensureSucceeds(await worldSocket.sendRequest({command: `create_entity: ego ${ego} controller=path_following_kmpc`}));
        egos[ego]._socket = new WebSocketAsPromised(`ws://localhost:8765/entity/${ego}`, WEBSOCKET_OPTIONS);
    }

    const worldCanvas = document.getElementById('worldCanvas');
    worldCanvas.getContext('2d').translate(worldCanvas.width/2, worldCanvas.height/2)

    setInterval(async () => {
        const worldState = ensureSucceeds(await worldSocket.sendRequest({command: 'tick'})).response;
        
        drawWorld(worldCanvas, worldState);
    }, 100);

    exported.tick = () => worldSocket.sendRequest({command: 'tick'}).then(console.log);
    exported.getState = () => worldSocket.sendRequest({command: 'state'}).then(console.log);
    exported.resetWorld = () => worldSocket.sendRequest({command: 'reset'}).then(console.log);
}

function drawWorld(canvas, worldState) {
    const ctx = canvas.getContext('2d');

    ctx.clearRect(-canvas.width/2, -canvas.height/2, canvas.width, canvas.height);

    for (const [entityName, entity] of Object.entries(worldState.entities)) {
        switch (entity.type) {
            case 'ego':
                drawVehicle(ctx, entity);
                break;
            default:
                console.error(`Unknown entity type "${entity.type}" for entity "${entityName}"`);
        }
    }
}

function drawVehicle(ctx, vehicle) {
    const vehicle_length = vehicle.L; // m
    const tireWidth = 0.2; // m
    const tireLength = 0.5; // m

    const vehicleState = decodeVehicleState(vehicle.state);

    const rearAxleCenter = [vehicleState.x, vehicleState.y];
    const frontAxleCenter = addvector(rearAxleCenter, [vehicle_length * Math.cos(vehicleState.theta), vehicle_length * Math.sin(vehicleState.theta)]);

    ctx.beginPath();
    // draw wheelbase
    ctx.moveTo(...rearAxleCenter.map(m_to_px));
    ctx.lineTo(...frontAxleCenter.map(m_to_px));

    // draw wheels
    ctx.save()
    ctx.translate(m_to_px(rearAxleCenter[0]), m_to_px(rearAxleCenter[1]));
    ctx.rotate(vehicleState.theta)
    ctx.rect(m_to_px(-tireLength/2), m_to_px(-tireWidth/2), m_to_px(tireLength), m_to_px(tireWidth));
    ctx.restore()

    ctx.save()
    ctx.translate(m_to_px(frontAxleCenter[0]), m_to_px(frontAxleCenter[1]));
    ctx.rotate(vehicleState.theta + vehicleState.delta)
    ctx.rect(m_to_px(-tireLength/2), m_to_px(-tireWidth/2), m_to_px(tireLength), m_to_px(tireWidth));
    ctx.restore()

    ctx.stroke();
}

main();

function decodeVehicleState(state) {
    return {
        x: state[0],
        y: state[1],
        theta: state[2],
        v: state[3],
        delta: state[4],
    }
}

// https://stackoverflow.com/a/42906936
function addvector(a, b) {
    return a.map((e, i) => e + b[i]);
}

function m_to_px(m) {
    // Meters to pixels
    return m * 20;
}
