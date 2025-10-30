const form = document.querySelector('form');

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const name = document.querySelector('#name').value;
  const email = document.querySelector('#email').value;
  const phone = document.querySelector('#phone').value;
  const date = document.querySelector('#date').value;

  // code to validate and process form data, and schedule an appointment
});
const navButton = document.querySelector('.nav-button');
const navMenu = document.querySelector('.nav-menu');

navButton.addEventListener('click', () => {
  navMenu.classList.toggle('visible');
});
