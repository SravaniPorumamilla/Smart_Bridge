// Rising Waters — form validation & light interactivity
document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('predict-form');
  if (!form) return;

  const rules = {
    annual_rainfall: { min: 0, max: 6000, label: 'Annual rainfall' },
    cloud_visibility: { min: 0, max: 20, label: 'Cloud visibility' },
    temperature: { min: -10, max: 55, label: 'Temperature' },
    humidity: { min: 0, max: 100, label: 'Humidity' },
    seasonal_rainfall: { min: 0, max: 4000, label: 'Seasonal rainfall' },
  };

  function validateField(input) {
    const rule = rules[input.name];
    const hint = document.querySelector(`.hint[data-for="${input.name}"]`);
    const value = parseFloat(input.value);

    if (input.value.trim() === '' || isNaN(value)) {
      input.classList.add('invalid');
      if (hint) { hint.textContent = `${rule.label} is required`; hint.classList.add('error'); }
      return false;
    }
    if (value < rule.min || value > rule.max) {
      input.classList.add('invalid');
      if (hint) {
        hint.textContent = `Expected between ${rule.min} and ${rule.max}`;
        hint.classList.add('error');
      }
      return false;
    }
    input.classList.remove('invalid');
    if (hint) { hint.textContent = ''; hint.classList.remove('error'); }
    return true;
  }

  form.querySelectorAll('input[data-validate]').forEach((input) => {
    input.addEventListener('blur', () => validateField(input));
    input.addEventListener('input', () => {
      if (input.classList.contains('invalid')) validateField(input);
    });
  });

  form.addEventListener('submit', function (e) {
    let allValid = true;
    form.querySelectorAll('input[data-validate]').forEach((input) => {
      if (!validateField(input)) allValid = false;
    });
    if (!allValid) {
      e.preventDefault();
      const firstInvalid = form.querySelector('.invalid');
      if (firstInvalid) firstInvalid.focus();
    }
  });
});
