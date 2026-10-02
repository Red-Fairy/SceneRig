export async function modelURL(source, signal) {
  if (!new URL(source, location.href).pathname.endsWith('.glb.gz')) return source;
  const response = await fetch(source, { signal });
  if (!response.ok) throw new Error(`Model download failed: ${response.status}`);
  let blob = await response.blob();
  const signature = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
  // Some hosts decode gzip through Content-Encoding before fetch returns it.
  if (signature[0] === 0x1f && signature[1] === 0x8b) {
    blob = await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).blob();
  }
  if (await blob.slice(0, 4).text() !== 'glTF') throw new Error('Invalid model data');
  signal?.throwIfAborted();
  return URL.createObjectURL(new Blob([blob], { type: 'model/gltf-binary' }));
}

export function releaseModelURL(source) {
  if (source?.startsWith('blob:')) URL.revokeObjectURL(source);
}
