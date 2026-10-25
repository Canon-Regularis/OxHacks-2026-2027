import { cpSync, existsSync, mkdtempSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
if (!existsSync(join(root, 'build/index.js'))) {
  console.error('Run npm run build before npm run serve:local.');
  process.exit(1);
}

// Serve an immutable build so checks and later builds cannot reload or remove
// the files being served to a presentation browser.
const snapshot = mkdtempSync(join(tmpdir(), 'eisengrau-local-'));
try {
  cpSync(join(root, 'build'), join(snapshot, 'build'), { recursive: true });
  symlinkSync(
    join(root, 'node_modules'),
    join(snapshot, 'node_modules'),
    'junction',
  );
  const host = process.env.HOST || '127.0.0.1';
  const port = process.env.PORT || '5174';
  const child = spawn(process.execPath, [join(snapshot, 'build/index.js')], {
    cwd: root,
    stdio: 'inherit',
    env: {
      ...process.env,
      HOST: host,
      PORT: port,
      ORIGIN: process.env.ORIGIN || `http://${host}:${port}`,
    },
  });
  for (const signal of ['SIGINT', 'SIGTERM'])
    process.on(signal, () => child.kill(signal));
  child.on('error', (error) => {
    console.error(error.message);
    rmSync(snapshot, { recursive: true, force: true });
    process.exitCode = 1;
  });
  child.on('exit', (code) => {
    rmSync(snapshot, { recursive: true, force: true });
    process.exitCode = code ?? 0;
  });
} catch (error) {
  rmSync(snapshot, { recursive: true, force: true });
  throw error;
}
