/// <reference lib="webworker" />
/// <reference lib="es2015" />

declare const self: ServiceWorkerGlobalScope;

const CACHE_NAME = 'crypto-trader-cache-v1';
const OFFLINE_URL = '/offline.html';

self.addEventListener('install', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll([
        '/',
        OFFLINE_URL,
        '/manifest.json',
        '/icons/icon-192x192.png',
        '/icons/icon-512x512.png'
      ]);
    })
  );
});

self.addEventListener('fetch', (event: FetchEvent) => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          return caches.match(OFFLINE_URL)
            .then(response => {
              if (response) return response;
              return caches.match(event.request);
            })
            .then(response => response || new Response('Network error', { status: 404 }));
        })
    );
    return;
  }

  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
      .catch(() => new Response('Network error', { status: 404 }))
  );
});

self.addEventListener('activate', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name.startsWith('crypto-trader-') && name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    })
  );
});

export {};
