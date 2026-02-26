# molrs-wasm: WASM bindings for molrs

Object-oriented WASM bindings for molrs using FFI handle-based architecture.

## Quick Start

```javascript
import init, { Frame, WasmArray, XyzReader } from './pkg/molrs.js';

await init();

// Create a new frame
const frame = new Frame();
const atoms = frame.createBlock("atoms");

// Set column data
atoms.setColumn("x", new Float32Array([1.0, 2.0, 3.0]));
const x = atoms.columnCopy("x");

// Zero-copy read view
const xView = atoms.columnView("x");
const xMem = xView.toTypedArray();

// Owned wasm array
const coords = WasmArray.from(new Float32Array([0, 0, 0, 1, 0, 0]), new Uint32Array([2, 3]));

// Read from file
const reader = new XyzReader(fileContent);
const loadedFrame = reader.read(0);
```

## API Reference

### Frame

The main container for molecular data.

```javascript
const frame = new Frame();

// Create a block
const atoms = frame.createBlock("atoms");

// Get an existing block
const atomsAgain = frame.getBlock("atoms");

// Remove a block
frame.removeBlock("atoms");

// Clear all blocks
frame.clear();

// Drop the frame (invalidates all blocks)
frame.drop();
```

### Block

A block contains columnar data with consistent row counts.

```javascript
const block = frame.createBlock("atoms");

// Set column data
const positions = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
const shape = new Uint32Array([2, 3]); // 2 atoms, 3 coordinates each
block.setColumn("positions", positions, shape);

// Get column data (returns copy)
const data = block.columnCopy("positions"); // Float32Array

// Get zero-copy view
const view = block.columnView("positions");
const zeroCopy = view.toTypedArray();

// Set from owned WasmArray
const owned = block.column("positions");
block.setFromView("positions_copy", owned);

// Query block info
const keys = block.keys(); // ["positions"]
const nrows = block.nrows(); // 2
```

### WasmArray

```javascript
const arr = new WasmArray(new Uint32Array([2, 3])); // zeros
arr.write_from(new Float32Array([1, 2, 3, 4, 5, 6]));
const zeroCopy = arr.toTypedArray();
const copied = arr.toCopy();
```

## File I/O

### Reading Files

```javascript
// XYZ files
const xyzReader = new XyzReader(fileContent);
const frameCount = xyzReader.len();
const frame = xyzReader.read(0); // Read frame 0

// PDB files
const pdbReader = new PdbReader(pdbContent);
const frame2 = pdbReader.read(0);

// LAMMPS data files
const lammpsReader = new LammpsReader(lammpsContent);
const frame3 = lammpsReader.read(0);
```

### Writing Files

```javascript
// TODO: Writer API needs implementation
// const output = writeFrame(frame, "xyz");
```

## TypeScript Example

```typescript
import init, { Block, Frame, XyzReader } from './pkg/molrs.js';

await init();

// Create frame and block
const frame: Frame = new Frame();
const atoms: Block = frame.createBlock("atoms");

// Set positions
const positions = new Float32Array([
  0.0, 0.0, 0.0,  // atom 1
  1.0, 0.0, 0.0,  // atom 2
  0.0, 1.0, 0.0,  // atom 3
]);
atoms.setColumn("x", positions);

// Query data
const nAtoms = atoms.nrows(); // 3
const x = atoms.columnCopy("x");
console.log(x[0]); // 0.0

const xView = atoms.columnView("x");
console.log(xView.isValid());

// Read from file
const reader = new XyzReader(fileContent);
const loadedFrame = reader.read(0);
if (loadedFrame) {
  const loadedAtoms = loadedFrame.getBlock("atoms");
  if (loadedAtoms) {
    console.log(`Loaded ${loadedAtoms.nrows()} atoms`);
  }
}
```

## Handle Invalidation

Handles follow strict invalidation rules:

- **Block handles** become invalid when:
  - The parent frame is dropped
  - The block is removed from the frame
  - The frame is cleared

- **Frame handles** become invalid when:
  - `frame.drop()` is called

Attempting to use an invalid handle will throw an error.

## Architecture

This WASM layer is a thin wrapper around `molrs-ffi`, which provides:
- Stable handle-based references (no raw pointers)
- Explicit invalidation semantics
- Consistent behavior across Python and WASM

The API is designed to be:
- **Object-oriented**: Natural class hierarchy (Frame → Block)
- **Type-safe**: Clear ownership and lifetime semantics
- **Explicit**: No hidden copies or implicit conversions
- **Simple**: Store is hidden - Frame is the main entry point

## Testing

```bash
cargo test -p molrs --lib
```

For WASM-specific tests:
```bash
wasm-pack test --node wasm
```
