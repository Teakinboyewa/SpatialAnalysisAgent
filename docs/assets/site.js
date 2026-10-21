// Small progressive enhancements: mobile menu, on-page contents, copy buttons, image lightbox.
(function () {
  "use strict";

  // Mobile navigation toggle
  var toggle = document.querySelector(".menu-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var open = document.body.classList.toggle("nav-open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  // "On this page" list built from h2 headings
  var toc = document.querySelector(".toc");
  var headings = document.querySelectorAll(".doc h2[id]");
  if (toc && headings.length >= 3) {
    var list = toc.querySelector("ol");
    headings.forEach(function (h) {
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      list.appendChild(li);
    });
    toc.hidden = false;
    if ("IntersectionObserver" in window) {
      var links = list.querySelectorAll("a");
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (l) {
            l.classList.toggle("active", l.getAttribute("href") === "#" + entry.target.id);
          });
        });
      }, { rootMargin: "0px 0px -70% 0px" });
      headings.forEach(function (h) { observer.observe(h); });
    }
  }

  // Copy buttons on code blocks
  document.querySelectorAll(".doc pre").forEach(function (pre) {
    if (!navigator.clipboard) return;
    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    button.addEventListener("click", function () {
      navigator.clipboard.writeText((pre.querySelector("code") || pre).innerText.trimEnd()).then(function () {
        button.textContent = "Copied";
        setTimeout(function () { button.textContent = "Copy"; }, 1600);
      });
    });
    pre.appendChild(button);
  });

  // Click a figure image to view it full size
  var dialog = document.querySelector(".lightbox");
  if (dialog && typeof dialog.showModal === "function") {
    var big = dialog.querySelector("img");
    var caption = dialog.querySelector(".lightbox-caption");
    document.querySelectorAll(".doc figure img").forEach(function (img) {
      img.tabIndex = 0;
      img.setAttribute("role", "button");
      img.setAttribute("aria-label", "Enlarge image: " + img.alt);
      function open() {
        big.src = img.currentSrc || img.src;
        big.alt = img.alt;
        var fc = img.closest("figure").querySelector("figcaption");
        caption.textContent = fc ? fc.textContent : "";
        dialog.showModal();
      }
      img.addEventListener("click", open);
      img.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
      });
    });
    dialog.addEventListener("click", function (e) { if (e.target === dialog) dialog.close(); });
  }
})();
