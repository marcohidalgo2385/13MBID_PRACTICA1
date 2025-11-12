from pathlib import Path
import joblib
import json

p = Path(__file__).parents[1] / 'models' / 'preprocessor.pkl'
print('preprocessor path:', p)
print('exists:', p.exists())
if not p.exists():
    raise SystemExit('preprocessor not found')
pre = joblib.load(p)
print('Preprocessor type:', type(pre))

# Try to inspect ColumnTransformer
if hasattr(pre, 'transformers_'):
    for name, trans, cols in pre.transformers_:
        print('\nTransformer:', name)
        print('Columns:', cols)
        print('Transformer type:', type(trans))
        # If pipeline
        if hasattr(trans, 'named_steps'):
            print('Named steps:', list(trans.named_steps.keys()))
            for step_name, step in trans.named_steps.items():
                print(' Step:', step_name, '->', type(step))
                if step_name.lower().startswith('onehot') or 'OneHotEncoder' in type(step).__name__:
                    ohe = step
                    print('  OneHotEncoder attributes:')
                    print('   drop =', getattr(ohe, 'drop', None))
                    print('   handle_unknown =', getattr(ohe, 'handle_unknown', None))
                    cats = getattr(ohe, 'categories_', None)
                    try:
                        print('   categories_ length:', None if cats is None else len(cats))
                        if cats is not None:
                            print('   sample categories for first 10 columns:')
                            for i, c in enumerate(cats[:10]):
                                print('    - col', i, ':', c[:20])
                    except Exception as e:
                        print('   error reading categories_', e)
        else:
            # Maybe direct OneHotEncoder
            if 'OneHotEncoder' in type(trans).__name__:
                ohe = trans
                print('  OneHotEncoder attributes:')
                print('   drop =', getattr(ohe, 'drop', None))
                print('   handle_unknown =', getattr(ohe, 'handle_unknown', None))
                print('   categories_ =', getattr(ohe, 'categories_', None))
else:
    print('Preprocessor has no transformers_ attribute; trying to inspect pipeline members...')

# Print feature_names_in_ if available
print('\nfeature_names_in_ =', getattr(pre, 'feature_names_in_', None))

# Also dump a json file for easy reading
out = {'preprocessor_type': str(type(pre)), 'has_transformers': hasattr(pre, 'transformers_')}
try:
    Path('tools').mkdir(exist_ok=True)
    with open('tools/inspect_preprocessor_output.json', 'w') as f:
        json.dump(out, f)
except Exception:
    pass

print('\nDone')
