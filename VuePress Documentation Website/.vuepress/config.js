module.exports = {
    title: 'Hello VuePress',
    description: 'Just playing around',
    themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Guide', link: '/guide/' },
          { text: 'External', link: 'https://google.com' },
        ],
        //sidebar: 'auto'
        sidebar: {
            '/guide/': [
                '',
                'More Informations'
            ]
        }
    }
}