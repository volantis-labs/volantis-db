import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import api from '../index.js';

test('E17 required public API', () => {
  const { rows, allowed_gaps } = JSON.parse(
    readFileSync(new URL('../../fixtures/required_api_v1.json', import.meta.url), 'utf8'),
  );
  assert.deepEqual(allowed_gaps.node, {}, 'E17 allows no new Node API gaps');
  const declarationPath = fileURLToPath(new URL('../index.d.ts', import.meta.url));
  const program = ts.createProgram([declarationPath], {
    noEmit: true, skipLibCheck: true, strict: true,
  });
  const declaration = program.getSourceFile(declarationPath);
  const checker = program.getTypeChecker();
  const exports = new Map(checker.getExportsOfModule(checker.getSymbolAtLocation(declaration))
    .map(symbol => [symbol.name, symbol]));

  function isPublic(declaration) {
    return !(ts.getCombinedModifierFlags(declaration) &
      (ts.ModifierFlags.Private | ts.ModifierFlags.Protected));
  }

  function hasPublicAPI(path) {
    const [name, member] = path.split('.');
    const symbol = exports.get(name);
    if (!symbol) return false;
    if (!(symbol.flags & ts.SymbolFlags.Value)) {
      return !member || !!checker.getPropertyOfType(checker.getDeclaredTypeOfSymbol(symbol), member);
    }
    if (!Object.hasOwn(api, name)) return false;
    if (!member) return true;

    const exportedType = checker.getTypeOfSymbolAtLocation(symbol, declaration);
    if (member === 'constructor') {
      return typeof api[name] === 'function' &&
        exportedType.getConstructSignatures().some(signature => isPublic(signature.getDeclaration()));
    }
    const staticProperty = checker.getPropertyOfType(exportedType, member);
    if (staticProperty) return isPublic(staticProperty.valueDeclaration) && member in api[name];
    const property = checker.getPropertyOfType(checker.getDeclaredTypeOfSymbol(symbol), member);
    if (!property || !isPublic(property.valueDeclaration)) return false;
    return !(property.flags & ts.SymbolFlags.Method) || typeof api[name].prototype[member] === 'function';
  }

  const missing = [];
  for (const [id, row] of Object.entries(rows)) {
    if (row.build === 'survey') continue;
    if (row.build === 'graph' && !api.Store.graphSupported()) continue;
    if (row.build === 'text' && (process.platform !== 'darwin' || process.arch !== 'arm64')) continue;
    if (!row.node?.length) missing.push(`${id}: no Node API specified`);
    for (const name of row.node ?? []) {
      if (!hasPublicAPI(name)) missing.push(`${id}: ${name}`);
    }
  }
  assert.deepEqual(missing, []);
});
