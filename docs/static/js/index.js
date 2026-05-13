window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})
document.addEventListener('DOMContentLoaded', function () {
	const carousel = bulmaCarousel.attach('#results-carousel', {
	  slidesToShow: 1,
	  slidesToScroll: 1,
	  loop: false,
	  autoplay: false,
	});
  
	const thumbnailContainer = document.querySelector('.thumbnail-container');
  
	if (carousel && thumbnailContainer) {
	  const items = carousel.items;
  
	  // 动态创建缩略图
	  items.forEach((item, index) => {
		const img = item.element.querySelector('img');
		if (!img) return;
  
		const thumb = document.createElement('img');
		thumb.src = img.src;
		thumb.alt = img.alt;
		thumb.classList.add('thumbnail');
		if (index === 0) thumb.classList.add('is-active');
  
		thumb.addEventListener('click', () => {
		  carousel.goTo(index);
		  updateThumbnails(index);
		});
  
		thumbnailContainer.appendChild(thumb);
	  });
  
	  // 更新当前缩略图高亮状态
	  function updateThumbnails(currentIndex) {
		const thumbs = thumbnailContainer.querySelectorAll('.thumbnail');
		thumbs.forEach((thumb, idx) => {
		  thumb.classList.remove('is-active');
		  if (idx === currentIndex) thumb.classList.add('is-active');
		});
	  }
  
	  // 监听轮播切换事件更新缩略图
	  carousel.on('before:change', (event) => {
		updateThumbnails(event.index);
	  });
	}
  });